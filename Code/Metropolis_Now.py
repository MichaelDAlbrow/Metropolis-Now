from __future__ import print_function

import sys
import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.ndimage.filters import maximum_filter
#from skimage.feature import peak_local_max


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import subplots_adjust

import corner


class Adaptive_Sampler(object):

	"""

	A class to implement an adaptive step-size Metropolis Hastings sampler.
	See Doran, M., Muller, C.M., Journal of Cosmology and Astroparticle Physics, 9, 3, 2004 for algorithm

	(c) Michael Albrow 2019

	inputs:

		nchains:				The number of chains torun in parallel. Must be at least 2.

		ndim:					The number of parameters of ln_prob_fn.

		ln_prob_fn:				A function that computes the natural log of the probability. Its first argument
								must be a numpy vector of length ndim.

		sigma:					Initial parameter uncertainties.

		initial_temperature: 	Thermodynamic temperature used to scale probability of acceptance of steps towards lower probability.
								If used, this would normally be increased as steps progress using log_temperature_change_rate
								in iterate_chains().

		args:					Extra positional arguments to pass through to ln_prob_fn.

		kwargs:					A dictionary of named arguments to pass through to ln_prob_fn.

		parameter_labels:		A list of strings to label parameters of ln_prob_fn when plotting.
	
	"""

	def __init__(self, ndim, ln_prob_fn, nchains=2, sigma=None, initial_temperature=1.0, args=None, kwargs=None, \
		parameter_labels=None, plot_to_file=True):

		if nchains < 2:
			raise ValueError("nchains must be at least 2.")
		self.nchains = nchains

		self.ndim = ndim

		self.parameter_labels = parameter_labels

		self.ln_prob_fn = _FunctionWrapper(ln_prob_fn, args, kwargs)

		self.ln_prob_fn_args = args
		self.ln_prob_fn_kwargs = kwargs

		self.chains = None
		self.ln_prob = None
		self.accepted_steps = None
		self.iterations = 0

		self.T = np.empty((nchains,ndim,ndim))
		self.sigma = np.empty((nchains,ndim))
		self.alpha = np.ones(nchains)
		self.R = 100.0*np.ones(ndim)
		self.temperature = initial_temperature

		self.alpha_history = []
		self.R_history = []
		self.steps_history = []

		for chain in range(nchains):
			self.T[chain] = np.eye(ndim)

		if sigma is None:
			for chain in range(nchains):
				self.sigma[chain] = np.ones(ndim)*1.e-6
		else:
			for chain in range(nchains):
				self.sigma[chain] = sigma
		
		self.plotprefix = 'plot'
		self.plot_to_file = plot_to_file

		self._iteration_calls = 0




	def iterate_chains(self,nsteps,start=None,plot_progress=False,min_diag_steps_ratio=10,scale_individual_chains=False, \
		use_latest_covariance=True,temperature_change_rate=1.e-3,plot_to_file=True):

		"""

		Advance chains and evaluate the Gelman-Rubin convergence diagnostic parameter R.

		nsteps:					The number of steps to advance

		start:					A 1-d numpy array of length ndim specifying mean starting position for all chains, or
								a 2-d numpy array of shape (nchains,ndim) specifying the starting points ofr each chain.

		plot_progress:			Plot chains and diagnostics as we go

		min_diag_steps_ratio:	Only diagonalise the covariance matrix for the proposal distribution
								if the number of accepted steps is greater than min_diag_steps_ratio
								times the number of dimensions.

		"""

		self._iteration_calls += 1

		# Add required storage space

		if self.chains is None and self._iteration_calls == 1:

			self.chains = np.empty((nsteps+1,self.nchains,self.ndim))
			self.ln_prob = np.empty((nsteps+1,self.nchains))
			self.accepted_steps = np.empty((nsteps+1,self.nchains))

			if start is None:
				start = np.zeros(ndim)

			for chain in range(self.nchains):

				if np.array(start).ndim == 1:
					self.chains[0,chain,:] = start + self.sigma[chain] * np.random.randn(self.ndim)
				else:
					self.chains[0,chain,:] = start[chain,:]

				self.ln_prob[0,chain] = self.ln_prob_fn(self.chains[0,chain,:])
				self.accepted_steps[0,chain] = 0

			self.iterations = 1

		else:

			self.grow_chains(nsteps)

		# Advance chains

		for step in range(nsteps):

			for chain in range(self.nchains):

				self.chains[self.iterations,chain,:], self.ln_prob[self.iterations,chain], self.accepted_steps[self.iterations,chain] = \
								self.mcmc_step(self.chains[self.iterations-1,chain],self.sigma[chain],self.T[chain],self.ln_prob[self.iterations-1,chain])

			self.iterations += 1


		# Discard points from the first call to this function

		if self._iteration_calls == 1:

			self.reset()

		else:


			# Compute Gelman-Rubin parameter

			w = np.zeros(self.ndim)
			m = np.zeros((self.nchains,self.ndim))
			cov = np.zeros((self.nchains,self.ndim,self.ndim))

			for chain in range(self.nchains):


				x = self.chains[-nsteps:,chain,:].reshape(nsteps,self.ndim)
				w += np.var(x,axis=0)
				m[chain,:] = np.mean(x,axis=0)

			w /= self.nchains

			b = np.float(nsteps)/(self.nchains-1) * np.sum( (m-np.mean(m,axis=0))**2 )
			self.R = ((w*(nsteps-1) + b)/nsteps)/w

			# Adjust acceptance factor

			if scale_individual_chains:

				for chain in range(self.nchains):

					a = np.sum(self.accepted_steps[-nsteps:,chain])/np.float(nsteps)
					if a > 0.4:
						self.alpha[chain] *= 1.1
					if a < 0.1:
						self.alpha[chain] /= 1.1
					self.alpha[chain] = np.max([self.alpha[chain],0.2])

			else:

				a = np.sum(self.accepted_steps[-nsteps:,:])/np.float(nsteps*self.nchains)

				if a > 0.4:
					self.alpha *= 1.1
				if a < 0.1:
					self.alpha /= 1.1
					
				if self.alpha[0] < 0.2:
					self.alpha = 0.2*np.ones(self.nchains)


			# Compute eigenvectors

			if use_latest_covariance:

				first = -nsteps
				length = nsteps

			else:

				first = 0
				length = self.iterations

			# Using combined chains
			ccov = np.cov(self.chains[first:,:,:].reshape(length*self.nchains,self.ndim),rowvar=False) * np.mean(self.alpha)
			cd, cT = np.linalg.eig(ccov)
			csd = np.sqrt(cd)


			for chain in range(self.nchains):

				if scale_individual_chains and (np.sum(self.accepted_steps[first:,chain]) > min_diag_steps_ratio*self.ndim):

					cov[chain] = np.cov(self.chains[first:,chain,:].reshape(length,self.ndim),rowvar=False) * self.alpha[chain]
					d, self.T[chain] = np.linalg.eig(cov[chain])
					self.sigma[chain] = np.sqrt(d)
						
				elif (np.sum(self.accepted_steps) > min_diag_steps_ratio*self.ndim):

					self.T[chain] = cT
					self.sigma[chain] = csd


			# Change thermodynamic temperature

			self.temperature -= temperature_change_rate * nsteps
			self.temperature = np.max([self.temperature,1.0])

			# Store record of convergence factors

			self.alpha_history.append(self.alpha.copy())
			self.R_history.append(self.R)
			self.steps_history.append(nsteps)

			if plot_progress:
				self.plot_chains(plot_to_file=plot_to_file)


	def grow_chains(self,nsteps):

		"""

		Add nsteps of extra storage space to chains.

		"""

		a = np.empty((nsteps,self.nchains,self.ndim))
		self.chains = np.concatenate((self.chains,a),axis=0) 

		a = np.empty((nsteps,self.nchains))
		self.ln_prob = np.concatenate((self.ln_prob,a),axis=0) 

		a = np.empty((nsteps,self.nchains))
		self.accepted_steps = np.concatenate((self.accepted_steps,a),axis=0) 


	def reset(self,retained_steps=1):

		"""
		Discard all save points in the chains.

		"""
		
		self.chains = self.chains[-retained_steps,:,:].reshape(retained_steps,self.nchains,self.ndim)
		self.ln_prob = self.ln_prob[-retained_steps,:].reshape(retained_steps,self.nchains)
		self.accepted_steps = self.accepted_steps[-retained_steps,:].reshape(retained_steps,self.nchains)
		self.iterations = retained_steps
		self.alpha_history = []
		self.R_history = []
		self.steps_history = []



	def mcmc_step(self,p,sigma_p,T,ln_prob_p=None):

		"""

		Advance chain by a single MCMC step.

		"""

		p1 = sigma_p * np.random.randn(self.ndim)

		p_try = p + np.dot(T,p1)

		if ln_prob_p is None:
			ln_prob_p = self.ln_prob_fn(p)

		ln_prob_try = self.ln_prob_fn(p_try)

		if (ln_prob_try > ln_prob_p) or (np.exp((ln_prob_try - ln_prob_p)/self.temperature) > np.random.rand()):

			return p_try, ln_prob_try, 1

		return p, ln_prob_p, 0



	def plot_chains(self,index=None,plot_lnprob=True,plot_alpha=True,plot_R=True,suffix='',parameter_labels=None,plot_to_file=True):


		"""

		Plot chains and diagnostics.

		"""

		if index is None:
			index = range(self.chains.shape[2])

		if parameter_labels is None:
			parameter_labels = self.parameter_labels

		if parameter_labels is None:
			parameter_labels = ["x[%d]"%i for i in range(self.ndim)]

		n_plots = len(index)
		extra_plots = np.sum([plot_lnprob,plot_alpha,plot_R])

		plt.figure(figsize=(8,11))
		
		subplots_adjust(hspace=0.0001)

		for i in range(n_plots):

			if i == 0:
				plt.subplot(n_plots+extra_plots,1,i+1)
				ax1 = plt.gca()
			else:
				plt.subplot(n_plots+extra_plots,1,i+1,sharex=ax1)

			ax = plt.gca()

			colours = plt.cm.rainbow(np.linspace(0,1,self.nchains))
			for chain in range(self.nchains):
				plt.plot(self.chains[:,chain,index[i]].T, '-', color=colours[chain], alpha=0.3)


			plt.ylabel(parameter_labels[i])

			if i < n_plots-1 + extra_plots:
				plt.setp(ax.get_xticklabels(), visible=False)
				ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
				ax.locator_params(axis='y',nbins=4)

		if plot_lnprob:

			plt.subplot(n_plots+extra_plots,1,n_plots+plot_lnprob,sharex=ax1)

			colours = plt.cm.rainbow(np.linspace(0,1,self.nchains))
			for chain in range(self.nchains):
				plt.plot(self.ln_prob[:,chain], '-', color=colours[chain], alpha=0.3)

			plt.ylabel(r"$ln P$")

			ax = plt.gca()

			if n_plots+plot_lnprob < n_plots-1 + extra_plots:
				plt.setp(ax.get_xticklabels(), visible=False)

			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)


		if plot_alpha:

			plt.subplot(n_plots+extra_plots,1,n_plots+plot_lnprob+plot_alpha,sharex=ax1)

			data = np.array(self.alpha_history)

			colours = plt.cm.rainbow(np.linspace(0,1,self.nchains))

			for chain in range(self.nchains):
				plt.plot(np.cumsum(np.array(self.steps_history)),data[:,chain], '-', color=colours[chain], alpha=0.3)


			plt.ylabel(r"$\alpha$")

			ax = plt.gca()

			if n_plots+plot_lnprob+plot_alpha < n_plots-1 + extra_plots:
				plt.setp(ax.get_xticklabels(), visible=False)

			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)


		if plot_R:

			plt.subplot(n_plots+extra_plots,1,n_plots+plot_lnprob+plot_alpha+plot_R,sharex=ax1)

			data = np.log10(np.array(self.R_history))

			colours = plt.cm.rainbow(np.linspace(0,1,self.ndim))

			for dim in range(self.ndim):
				plt.plot(np.cumsum(np.array(self.steps_history)),data[:,dim], '-', color=colours[dim], alpha=0.3)

			plt.ylabel(r"$\log_{10} R$")

			ax = plt.gca()

			if n_plots+plot_lnprob+plot_alpha < n_plots-1 + extra_plots:
				plt.setp(ax.get_xticklabels(), visible=False)

			ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
			ax.locator_params(axis='y',nbins=4)

		if plot_to_file:
			plt.savefig(self.plotprefix+'-chain'+suffix)
			plt.close()


	def plot_chain_corner(self,chain=None,nsteps=None,parameter_labels=None,corner_kwargs=None,plot_to_file=True):

		"""

		Make corner plots and histograms for the parameter pairs.

		"""

		if nsteps is None:
			nsteps = self.chains.shape[0]

		if parameter_labels is None:
			parameter_labels = ["x[%d]"%i for i in range(self.ndim)]

		if chain is None:
			data = self.chains[-nsteps:,:,:].reshape(nsteps*self.nchains,self.ndim)
		else:
			data = self.chains[-nsteps:,chain,:]

		if corner_kwargs is None:
			corner_kwargs = dict()

		figure = corner.corner(data,
					labels=parameter_labels,
					quantiles=[0.16, 0.5, 0.84],
					show_titles=True, title_args={"fontsize": 12}, **corner_kwargs)

		if plot_to_file:
			figure.savefig(self.plotprefix+'-pdist.png')



	def detect_multiple_solutions(self,smoothing_width=1.0):

		samples = self.chains.reshape(self.iterations*self.nchains,self.ndim)
		kde_skl = KernelDensity(kernel='gaussian',bandwidth=smoothing_width)
		kde_skl.fit(samples)



class _FunctionWrapper(object):

    """

    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.

    Stolen from DFM's emcee code.

    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:  # pragma: no cover
            import traceback

            print("Adaptive_Sampler: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


