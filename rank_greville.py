#!/usr/bin/env python3

# Import librairies
import numpy as np
import math
import warnings
import copy

class RecursiveModel(object):
	"""Rank-decomposed linear model with recursive capabilities."""
	###################
	#    Notations    #
	###################
	# X		<=> self.parameters
	# Y		<=> self.target_values
	# A		<=> self.observations
	# B		<=> self.decomposed_observations
	# C		<=> self.observations_basis
	# C̃		<=> self.dual_observations_basis = dual(C) = inv(σ).C = inv(C.tr(C)).C
	# P⁻¹		<=> self.sum_proj_inv = inv(P) = inv(tr(B).B)
	# A⁺		<=> self.pseudo_inverse = pinv(A) = pinv(B.C) = tr(dual(C)).inv(P).tr(B))
	# Γ		<=> self._new_observation (new observation)
	# γ		<=> self.decomp_obs (decompositon of new observation in the observations basis) = dual(C).Γ
	# Γ'		<=> self.obs_reject (rejection vector of new_observation with respect to the observations basis) = Γ - tr(C).γ
	# ζ		<=> self.proj_obs = inv(P).γ = inv(P).dual(C).Γ
	# β		<=> self.pinv_obs = B.ζ = B.inv(P).γ = B.inv(P).dual(C).Γ = tr(pinv(A)).Γ
	# Var(X)	<=> self.variance_parameters
	# nVar(X)_empiric	<=> self.norm_empiric_variance = Var(x)_empiric/Var_norm(err)
	# Var(X)_guess	<=> self.guess_variance
	# Var(g)	<=> self.variance_parameters_guess
	# Var_norm(err) <=> var_noise (empirical estimation of the variance of the noise) = sum(err**2)/(n-r) (= s**2)
	# err		<=> err (residuals) = Y - A.X
	# X'		<=> X' (parameters of the purely overdetermined system, i.e. B.X' = Y) = pinv(B).Y
	# σ		<=> sigma (Gram matrix of observations basis) = C.tr(C)
	# n		= self.nb_obs (number of observations)
	# r		= self.nb_core_obs = |observations basis| = rank of A
	# m		= self.nb_regressors (number of regressors) 
	
	def __init__(self, eps=None, pseudo_inverse_update=False, covariance_update=False, pseudo_inverse_support=False, covariance_support=False, full_storage=False, empirical_variance=False):
		# Model parameters
		self.parameters = np.empty(0) # Parameters of the model, i.e. least-square solution (called X)
		
		# Raw observations storage
		self.target_values = np.empty(0) # Observed values for the regressand (called Y)
		self.observations = np.empty((0,0)) # Observations for the regressors (called A = B.C)
		self.observations_basis = np.empty((0,0)) # Basis of observations (called C)
		self.decomposed_observations = np.empty((0,0)) # Observations decomposed in the basis C (called B)
		
		# Precomputed items
		self.dual_observations_basis = np.empty((0,0)) # Dual of the observations basis (called dual(C) = C̃ = inv(σ).C = inv(C.tr(C)).C)
		self.sum_proj_inv = np.empty((0,0)) # Inverse of the sum of projectors P = tr(B).B (called inv(P) = inv(tr(B).B))
		self.pseudo_inverse = np.empty((0,0)) # Full pseudo-inverse (called pinv(A) = pinv(B.C) = tr(dual(C)).inv(P).tr(B))
		self._purge_temp_storage() # Temporary storage of precomputed vectors Γ, γ, Γ', ζ, β
		
		# Covariance-related storage
		self.variance_parameters = np.empty((0,0)) # Model parameters covariance matrix (called Var(X))
		self.norm_empiric_variance = np.empty((0,0)) # Normalized empirical contribution to Var(x) (called nVar(X)_empiric)
		self.guess_variance = np.empty((0,0)) # Guessed contribution to Var(X) (called Var(X)_guess)
		self.variance_parameters_guess = np.empty((0,0)) # Guessed parameters covariance matrix (called Var(g))
		
		# Linear dependancy threshold
		self.eps = eps
		
		# Size counters
		self.nb_obs = 0 # Total number of observations (called n)
		self.nb_core_obs = 0 # Number of linearly independant observations (i.e. rank of A, called r)
		self.nb_regressors = 0 # Number of regressors (called m)
		
		# Algorithms variants
		self.covariance_update = covariance_update # Store additional data and perform covariance matrix update
		self.pseudo_inverse_update = pseudo_inverse_update # Store additional data and perform the pseudo-inverse matrix update
		self.covariance_support = covariance_support # Store additional data required for computing covariance matrix (no update, just storage)
		self.pseudo_inverse_support = pseudo_inverse_support # Store additional data required for computing pseudo-inverse (no update, just storage)
		self.full_storage = full_storage # Store all possible data
		self.empirical_variance = empirical_variance # Uniformly estimate Var(g) from the variance of the noise Var_norm(err), i.e. Var(g) = 1*Var_norm(err)
	
	#################################
	#    Formulae implementation    #
	#################################
	
	#
	# Linear algebra details can be found on ...
	#
	# Here, we are considering a general non-orthogonal rank factorisation updating scheme.
	
	def add_observations(self, new_observations, new_values):
		"""Update the model by adding a new observation.
		
		Parameters:
			self (Model object): Model object to update.
			
			new_observations (2D array or iterable of lists): New observations for the regressors (x_1, ..., x_m).
				Notes: expected len(new_observations) = n
			
			new_values (1D array or list): Associated values for the regressand.
				Notes: expected len(new_values) = n
		
		Returns:
			parameters_update (1D array)
		"""
		
		initial_parameters = self.parameters.copy()
		
		for observation, value in zip(new_observations, new_values):
			self.add_observation(observation, value)
		
		tmp_array = np.zeros(self.parameters.shape, dtype=self.parameters.dtype)
		tmp_array[:initial_parameters.shape[0]] = initial_parameters
		initial_parameters = tmp_array
		
		return(self.parameters - initial_parameters)
	
	def add_observation(self, new_observation, new_value):
		"""Update the model by adding a new observation.
		
		Parameters:
			self (Model object): Model object to update.
			
			new_observation (1D array or list): New observation for the regressors (x_1, ..., x_m).
			
			new_value (float): Associated value for the regressand.
		
		Returns:
			parameters_update (1D array)
		"""
		
		# Ensure internal new_observation is a Numpy array
		self._new_observation = np.array(new_observation)
		
		# Update number of regressors to fit new observation
		if self._new_observation.shape[0] > self.nb_regressors:
			self.add_new_regressors(self._new_observation.shape[0] - self.nb_regressors)
		
		# Compute if new observation is redundant with previous observations (i.e. can be written as linear combination of observations from core_observations)
		# Cost: time: O(mr) ; space: O(mr)
		redundant = self._evaluate_redundancy()
		
		# Compute Kalman gain vector
		# Cost: time: O(mr) or O(m) ; space: O(m)
		kalman_gain = self._compute_kalman_gain(redundant)
		
		# Update least-squares solution
		# Cost: time: O(m) ; space: O(m)
		parameters_update = self._update_parameters(kalman_gain, new_value)
		
		# Update pseudo-inverse, if requested
		# Cost: time: O(mn) ; space: O(mn)
		if self.pseudo_inverse_update:
			self._update_pseudo_inverse(kalman_gain)
		
		# Update model covariance matrix components, if requested
		# Cost: time: O(m²) ; space: O(m²)
		if self.covariance_update:
			if redundant:
				self._dupl_update_covariance_components(kalman_gain)
			else:
				self._core_update_covariance_components(kalman_gain)
		
		# Update model storage
		# Cost: time: O(mr) ; space: O(mr) (if full storage is requested, time and space complexity are O(mr+nr))
		if redundant:
			self._dupl_update_model_storage(new_value)
		else:
			self._core_update_model_storage(kalman_gain, new_value)
		
		# Update covariance matrix (with now fully updated model), if requested
		# Cost: time: O(m²) ; space: O(m²)
		if self.covariance_update:
			self._update_covariance()
		
		# Purge temporary storage of precomputed data
		self._purge_temp_storage()
		
		return(parameters_update)
	
	def _evaluate_redundancy(self, eps=None, obs_reject=None):
		"""Returns whether the current new observation is a linear combination of previous observations
		
		Computational details:
			The current new observation is considered a linear combination of previous observations if the norm of its associated rejection vector is lower than `self.eps` (absolutely or relatively to the new observation)."""
		
		if obs_reject is None:
			# Use internally computed Γ' vector
			# Cost: time: O(mr) ; space: O(m)
			obs_reject = self.obs_reject
		
		if eps is None:
			# Estimate eps from cumulative floating point error (for computing Γ'), if requested
			if self.eps is None:
				# Retrieve machine precision for rejection vector representation
				try:
					eps = np.finfo(obs_reject.dtype).eps
				except ValueError: # Rejection vector is in exact representation
					return(np.all(obs_reject == 0))
				
				# Rescale machine precision to account for cumulative rounding error during rejection vector computation
				eps *= (self.nb_regressors**2*self.nb_core_obs
				        + self.nb_regressors*self.nb_core_obs
				        + self.nb_regressors)
			else:
				eps = self.eps
		
		# Check if rejection vector is small enough (smaller than predefined relative cutoff)
		# Cost: time: O(m) ; space: O(1)
		return(np.dot(obs_reject, obs_reject) < eps**2*max(1, np.dot(self._new_observation, self._new_observation)))
	
	def _compute_kalman_gain(self, redundant, decomp_obs=None, proj_obs=None, obs_reject=None):
		"""Compute Kalman gain vector associated with new observation.
		
		Parameters:
			self (Model object): Model object.
			
			redundant (boolean): Linear dependancy of current new observation from previous observations.
		
		Returns:
			kalman_gain (1D array)
		"""
		
		if redundant:
			if decomp_obs is None:
				# Use internally computed γ vector
				# Cost: None (already computed)
				decomp_obs = self.decomp_obs
			
			if proj_obs is None:
				# Use internally computed ζ vector
				# Cost: time: O(mr) ; space: O(r)
				proj_obs = self.proj_obs
			
			# Compute rescaled proj_obs: ζ/(1+tr(γ).ζ)
			# Cost: time: O(r) ; space: O(r)
			rescaled_proj_obs = proj_obs / (1 + (np.dot(decomp_obs, proj_obs) or 0)) # default value 0 if tr(γ).ζ is None
			
			# Compute Kalman gain vector: tr(dual(C)).ζ/(1+tr(γ).ζ)
			# Cost: time: O(mr) ; space: O(m)
			kalman_gain = np.dot(self.dual_observations_basis.T, rescaled_proj_obs)
		else:
			if obs_reject is None:
				# Use internally computed Γ' vector
				# Cost: None (already computed)
				obs_reject = self.obs_reject
			
			# Compute the associated Schur complement S = ||Γ'||²
			# Cost: time: O(m) ; space: O(1)
			schur_complement = np.dot(obs_reject, obs_reject)
			
			# Compute Kalman gain vector: Γ'/||Γ'||²
			# Cost: time: O(m) ; space: O(m)
			kalman_gain = obs_reject / schur_complement
		
		return(kalman_gain)
	
	def _update_parameters(self, kalman_gain, new_value, new_observation=None):
		"""Update linear model parameters
		
		Parameters:
			self (Model object): Model object.
			
			kalman_gain (1D array): Associated Kalman gain.
			
			new_value (float): Associated target value for the current new observation.
		
		Returns:
			parameters_update (1D array)
		"""
		
		if new_observation is None:
			new_observation = self._new_observation
		
		# Compute predicted residual associated with new observation
		# Cost: time: O(m) ; space: O(1)
		predicted_residual = new_value - np.dot(self.parameters, new_observation)
		
		# Compute least-squares update
		# Cost: time: O(m) ; space: O(m)
		try:
			parameters_update = kalman_gain * predicted_residual
		except TypeError:
			kalman_gain[kalman_gain == None] = 0
			parameters_update = kalman_gain * predicted_residual
		
		# Update model parameters
		# Cost: time: O(m) ; space: O(1)
		self.parameters += parameters_update
		
		return(parameters_update)
	
	def _update_pseudo_inverse(self, kalman_gain, pinv_obs=None):
		"""Update pseudo-inverse
		
		Parameters:
			self (Model object): Model object.
			
			kalman_gain (1D array): Associated Kalman gain.
		"""
		
		# Ensures required storage is enabled
		assert(self.decomposed_observations_storage)
		
		if pinv_obs is None:
			# Use internally computed β vector
			# Cost: time: O(nr) ; space: O(n) (+ ζ computation cost if non redundant case: time: O(mr) ; space: O(r))
			pinv_obs = self.pinv_obs
		
		# Update pseudo-inverse: pinv(A_new) = (pinv(A) - kalman_gain.tr(β) | kalman_gain)
		# Cost: time: O(mn) ; space: O(mn)
		tmp_array = np.empty((self.nb_regressors, self.nb_obs+1), dtype=kalman_gain.dtype)
		tmp_array[:, :-1] = self.pseudo_inverse - np.outer(kalman_gain, pinv_obs)
		tmp_array[:, -1] = kalman_gain
		self.pseudo_inverse = tmp_array
	
	def _core_update_covariance_components(self, kalman_gain, obs_reject=None, decomp_obs=None, proj_obs=None):
		"""Update the model variance components (empiric and guessed) by adding a new configuration that is linearly independant with previous configurations.
		
		Parameters:
			self (Model object): Model object to update.
			
			kalman_gain (1D array): Associated Kalman gain: Γ'/||Γ'||².
		
		Computational details:
			See help(self.scratch_covariance_computation) for more info on variance computation
			
			Var(x) is decomposed into:
			Var(x)_new = Var(x)_guess_new + Var_norm(err_new)*nVar(x)_empiric_new
			
			Using a decomposed form for dual(C_new), we have:
			nVar(x)_empiric_new = tr(dual(C_new)).inv(P_new).dual(C_new) = (tr(dual(C)) - kalman_gain.tr(γ) | kalman_gain).(inv(P)/0|0/1).tr(...) = nVar(x)_empiric - (tr(dual(C)).ζ.tr(kalman_gain) + tr(...)) + kalman_gain.tr(kalman_gain)*(1+tr(γ).ζ)
			
			Also, tr(dual(C_new)).C_new can be rewritten: (tr(dual(C)) - kalman_gain.tr(γ) | kalman_gain).(C/tr(Γ)) = tr(dual(C)).C + (kalman_gain.tr(Γ) - kalman_gain.tr(γ).C) = tr(dual(C)).C + kalman_gain.tr(Γ'). So one can write:
			Var(X)_guess_new = Var(g - tr(dual(C_new)).C_new.g) = (1 - tr(dual(C_new)).C_new).Var(g).tr(1 - tr(dual(C_new)).C_new) = Var(X)_guess - ((1-tr(dual(C)).C).Var(g).kalman_gain.tr(Γ') + tr(...)) + kalman_gain.tr(kalman_gain)*(tr(Γ').Var(g).Γ')
			(Assuming Var(g) is symmetric, which it should be...)
		
		Note:
			This method only updates nVar(X)_empiric and Var(X)_guess, not Var(X).
			
			The computation of Var(X) (merging of variance components) is performed by self._update_covariance, since the variance of the noise should be computed on fully updated model.
		"""
		
		if obs_reject is None:
			# Use internally computed Γ' vector
			# Cost: None (already computed)
			obs_reject = self.obs_reject
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		if proj_obs is None:
			# Use internally computed ζ vector
			# Cost: time: O(mr) ; space: O(r) (already computed if pseudo-inverse update)
			proj_obs = self.proj_obs
		
		# Compute -nVar(X)_empiric update
		# Total cost: time: O(m²) ; space: O(m²)
		if self.nb_core_obs > 0:
			# First part (1st order kalman_gain factor: tr(dual(C)).ζ.tr(kalman_gain) + tr(...))
			# Cost: time: O(mr+m²) ; space: O(m+m²)
			variance_update = np.outer(np.dot(self.dual_observations_basis.T, proj_obs), kalman_gain)
			variance_update = variance_update + variance_update.T
			# Second part (2nd order kalman_gain factor: kalman_gain.tr(kalman_gain)*(1+tr(γ).ζ))
			# Cost: time: O(r+m+m²) ; space: O(m+m²)
			variance_update -= np.outer(kalman_gain, kalman_gain*(1 + np.dot(decomp_obs, proj_obs)))
		else:
			# First recording
			# Cost: time: O(m²) ; space: O(m²)
			variance_update = -np.outer(kalman_gain, kalman_gain)
		
		# Update nVar(X)_empiric
		self.norm_empiric_variance -= variance_update
		
		# Compute -Var(X)_guess update
		# Total cost: time: O(m²) ; space: O(m²)
		# First part (1st order kalman_gain factor: (1-tr(dual(C)).C).Var(g).kalman_gain.tr(Γ') + tr(...))
		# Cost: time: O(mr+m²) ; space: O(m+m²)
		if self.empirical_variance: # XXX: In such case, the variance components decomposition is not optimal (major further optimisations can be achieved when combining them (i.e. merging outer products))
			variance_update = kalman_gain # Use Var(g) = 1
		else:
			variance_update = np.dot(self.variance_parameters_guess, kalman_gain) # Var(g).kalman_gain
		variance_update -= np.dot(self.dual_observations_basis.T, np.dot(self.observations_basis, variance_update)) # (1-tr(dual(C)).C).Var(g).kalman_gain
		variance_update = np.outer(variance_update, obs_reject)
		variance_update += variance_update.T
		# Second part (2nd order kalman_gain factor: kalman_gain.tr(kalman_gain)*(tr(Γ').Var(g).Γ'))
		# Cost: time: O(m+m²) ; space: O(m+m²)
		if self.empirical_variance:
			tmp_factor = np.dot(obs_reject, obs_reject) # Use Var(g) = 1
		else:
			tmp_factor = np.dot(obs_reject, np.dot(self.variance_parameters_guess, obs_reject))
		variance_update -= np.outer(kalman_gain, kalman_gain*tmp_factor)
		
		# Update Var(X)_guess
		self.guess_variance -= variance_update
	
	def _dupl_update_covariance_components(self, kalman_gain, decomp_obs=None, proj_obs=None):
		"""Update the model variance components (empiric and guessed) by adding a new configuration that is a linear combination of previous configurations.
		
		Parameters:
			self (Model object): Model object to update.
			
			kalman_gain (1D array): Associated Kalman gain.
		
		Computational details:
			See help(self.scratch_covariance_computation) for more info on variance computation
			
			Var(X) is decomposed into:
			Var(X)_new = Var(X)_guess_new + Var_norm(err_new)*nVar(X)_empiric_new
			
			Using a decomposed form for inv(P_new), we have:
			nVar(X)_empiric_new = tr(dual(C_new)).inv(P_new).dual(C_new) = tr(dual(C)).(inv(P)-ζ.tr(ζ)/(1+tr(γ).ζ)).dual(C) = nVar(X)_empiric - tr(dual(C)).ζ.tr(ζ).dual(C)/(1+tr(γ).ζ)
			
			Since C is not modified, the guessed variance component does not change:
			Var(X)_guess_new = Var(g - tr(dual(C)).C.g) = Var(X)_guess
		
		Note:
			This method only updates nVar(X)_empiric and Var(X)_guess, not Var(X).
			
			The computation of Var(X) (merging of variance components) is performed by self.update_covariance, since the variance of the noise should be computed on fully updated model.
		"""
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		if proj_obs is None:
			# Use internally computed ζ vector
			# Cost: None (already computed)
			proj_obs = self.proj_obs
		
		# Compute -nVar(X)_empiric update
		# Total cost: time: O(m²) ; space: O(m²)
		# Equation: tr(dual(C)).ζ.tr(ζ).dual(C)/(1+tr(γ).ζ)
		# Cost: time: O(mr+r+m+m²) ; space: O(m+m²)
		variance_update = np.dot(self.dual_observations_basis.T, proj_obs)
		variance_update = np.outer(variance_update, variance_update/(1 + np.dot(decomp_obs, proj_obs)))
		
		# Update nVar(X)_empiric
		self.norm_empiric_variance -= variance_update
	
	def _update_covariance(self):
		"""Compute the covariance matrix for the contribution terms of the model.
		
		Parameters:
			self (Model object): Model object to update.
		
		Computational details:
			See help(self.scratch_covariance_computation) for more info on variance computation
			
			Var(X) is decomposed into:
			Var(X)_new = Var(X)_guess_new + Var_norm(err_new)*nVar(X)_empiric_new
		"""
		# Ensures required storage is enabled
		assert(self.observations_storage)
		
		# Check if variance-related empirical data is available
		if self.nb_obs > self.nb_core_obs:
			# Update the model covariance matrix
			if self.empirical_variance:
				# In this case, Var(g) is taken as 1*Var_norm(err), and Var(X)_guess_new was computed with Var(g) = 1
				# Therefore, Var(X)_new = Var_norm(err_new)*(Var(X)_guess_new + nVar(X)_empiric_new)
				# Cost: time: O(m²) ; space: O(m²)
				self.variance_parameters = self.var_noise*(self.guess_variance + self.norm_empiric_variance)
			else:
				# Var(X)_new = Var(X)_guess_new + Var_norm(err_new)*nVar(X)_empiric_new
				# Cost: time: O(m²) ; space: O(m²)
				self.variance_parameters = self.guess_variance + self.var_noise*self.norm_empiric_variance
		else:
			# In that case, the variance of the noise cannot be estimated from empirical data.
			# Estimating Var(X) from guess leads to:
			self.variance_parameters = self.variance_parameters_guess # In other words, there is simply no empirical contribution to the variance of the model parameters.
	
	def _core_update_model_storage(self, kalman_gain, new_value, new_observation=None, decomp_obs=None):
		"""Update linear model parameters by adding a new configuration that is linearly independant with previous configurations.
		
		Parameters:
			self (Model object): Model object.
			
			kalman_gain (1D array): Associated Kalman gain.
			
			new_value (float): Associated target value for the current new observation.
		"""
		
		if new_observation is None:
			new_observation = self._new_observation
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		# Update observation basis: C_new = (C/tr(Γ))
		# Cost: time: O(mr) ; space: O(mr)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_regressors), order='F', dtype=kalman_gain.dtype)
		tmp_array[:-1] = self.observations_basis
		tmp_array[-1] = new_observation
		self.observations_basis = tmp_array
		
		# Update dual of observations basis: dual(C_new) = (dual(C) - γ.tr(kalman_gain)/tr(kalman_gain))
		# Cost: time: O(mr) ; space: O(mr)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_regressors), dtype=kalman_gain.dtype)
		tmp_array[:-1] = self.dual_observations_basis - np.outer(decomp_obs, kalman_gain)
		tmp_array[-1] = kalman_gain
		self.dual_observations_basis = tmp_array
		
		# Update inv(P) matrix: inv(P_new) = (inv(P)/0|0/1)
		# Cost: time: O(r²) ; space: O(r²)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
		tmp_array[:-1, :-1] = self.sum_proj_inv
		tmp_array[:-1, -1] = tmp_array[-1, :-1] = 0
		tmp_array[-1, -1] = 1
		self.sum_proj_inv = tmp_array
		
		if self.decomposed_observations_storage:
			# Update decomposed observations: B_new = (B/0|0/1)
			# Cost: time: O(nr) ; space: O(nr)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
			tmp_array[:-1, :-1] = self.decomposed_observations
			tmp_array[:-1, -1] = tmp_array[-1, :-1] = 0
			tmp_array[-1, -1] = 1
			self.decomposed_observations = tmp_array
		
		if self.observations_storage:
			# Update raw observations: A_new = (A/tr(Γ))
			# Cost: time: O(mn) ; space: O(mn)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_regressors), dtype=kalman_gain.dtype)
			tmp_array[:-1] = self.observations
			tmp_array[-1] = new_observation
			self.observations = tmp_array
			
			# Update regressand target values:
			self.target_values = np.append(self.target_values, new_value)
		
		# Update internal counters
		self.nb_core_obs += 1
		self.nb_obs += 1
	
	def _dupl_update_model_storage(self, new_value, new_observation=None, decomp_obs=None, proj_obs=None):
		"""Update linear model parameters by adding a new configuration that is a linear combination of previous configurations.
		
		Parameters:
			self (Model object): Model object.
			
			new_value (float): Associated target value for the current new observation.
		"""
		
		if new_observation is None:
			new_observation = self._new_observation
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		if proj_obs is None:
			# Use internally computed ζ vector
			# Cost: None (already computed)
			proj_obs = self.proj_obs
		
		# Update observation basis: C_new = C
		# Nothing to do...
		
		# Update dual of observations basis: dual(C_new) = dual(C)
		# Nothing to do...
		
		# Update inv(P) matrix: inv(P_new) = inv(P) - ζ.tr(ζ)/(1+tr(γ).ζ)
		# Cost: time: O(r²) ; space: O(r²)
		self.sum_proj_inv = self.sum_proj_inv - np.outer(proj_obs, proj_obs/(1 + (np.dot(decomp_obs, proj_obs) or 0))) # default value 0 if tr(γ).ζ is None
		
		if self.decomposed_observations_storage:
			# Update decomposed observations: B_new = (B/tr(γ))
			# Cost: time: O(nr) ; space: O(nr)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_core_obs), dtype=decomp_obs.dtype)
			tmp_array[:-1] = self.decomposed_observations
			tmp_array[-1] = decomp_obs
			self.decomposed_observations = tmp_array
		
		if self.observations_storage:
			# Update raw observations: A_new = (A/tr(Γ))
			# Cost: time: O(mr) ; space: O(mr)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_regressors), dtype=decomp_obs.dtype)
			tmp_array[:-1] = self.observations
			tmp_array[-1] = new_observation
			self.observations = tmp_array
			
			# Update regressand target values:
			self.target_values = np.append(self.target_values, new_value)
		
		# Update internal counters
		self.nb_obs += 1
	
	def scratch_covariance_computation(self):
		"""Compute the covariance matrix for the contribution terms of the model, from scratch.
		
		Parameters:
			self (Model object): Model object to update.
		
		Returns:
			variance_parameters (2D array)
		
		Computational details:
			For the overdetermined system B.X' = Y, the covariance matrix of the least-square estimator X̂' is computed as Var(X̂'|B) = inv(tr(B).B)*Var_norm(err), where Var_norm(err) is the variance of the noise, and err = Y - Ŷ = Y - B.X̂' (beware, the variance of the noise is based on err but is not the variance of err in the general case (i.e. Var_norm(err) != Var(err)), due to degree of freedom considerations). (Bibliography: Greene, Econometric Analysis)
			
			For the full system though A.X = B.C.X = Y, this covariance is not necessarily defined for the whole regressors space, since it does not apply to the kernel of C. Indeed, the pseudo-inverse of A will only spawn on Im(tr(C)) (i.e. the space of the actually observed variables).
			This is coherent with the fact that no data is available for the unobserved space (i.e. Ker(C)), and therefore no empirical covariance can be found for this sub-space.
			
			However, one can use arbitrary covariance on this kernel, to replace empirical data.
			This can be done by projecting the guessed covariance (that spawns on all regressors) onto the kernel of C.
			Such guessed variance can be written Var(g) = tr(C).a + Var(g)_ker, since Im(tr(C)) + Ker(C) = E, where E is the regressors ambient space.
			With this decomposition, it is easy to verify that P_Ker(C) = 1 - P_Im(tr(C)) = 1 - tr(C).inv(C.tr(C)).C = 1 - tr(C).dual(C) is a projector onto Ker(C).
			
			It comes finally that the variances on the initial variables can be written:
			Var(X) = Var(P_Im(tr(C)).X + P_Ker(C).X) = Var(X)_empiric + Var(X)_guess (seen and unseen variables are uncorrelated)
			
			Var(X)_empiric = Var(pinv(A).Y) = Var(tr(dual(C)).X̂') = tr(dual(C)).Var(X̂').dual(C)
			Var(X)_guess = Var(g - tr(dual(C)).C.g) = (1 - tr(dual(C)).C).Var(g).tr(1 - tr(dual(C)).C)
			
			Combining all equations, one has the following final form for the variances on the initial variables:
			Var(x) = tr(dual(C)).(inv(P)*Var_norm(err)).dual(C) + (1 - tr(dual(C)).C).Var(g).(1 - tr(C).dual(C))
			
			Note: One can also estimate Var(g) from Var_norm(err) as Var(g) = 1*Var_norm(err).
			This behavior can be enabled with the self.empirical_variance flag.
		"""
		
		# Check if variance-related empirical data is available
		if self.nb_obs > self.nb_core_obs:
			# Update guessed model parameters covariance, if applicable
			if self.empirical_variance:
				# Uniformly update the guessed variances on the initial variables Var(g) from the empirical variance of the noise Var_norm(err)
				self.variance_parameters_guess = np.eye(self.nb_regressors)*self.var_noise
			
			# Compute the variances on the initial variables
			# Var(X)_empiric = Var(tr(dual(C)).X̂') = tr(dual(C)).Var(X̂').dual(C)
			self.variance_parameters = np.dot(np.dot(self.dual_observations_basis.T, self.sum_proj_inv*self.var_noise), self.dual_observations_basis)
			# Var(X)_guess = (1 - tr(dual(C)).C).Var(g).tr(1 - tr(dual(C)).C) = (tr(dual(C)).C - 1).Var(g).tr(tr(dual(C)).C - 1)
			tmp_array = np.dot(self.dual_observations_basis.T, self.observations_basis)
			np.fill_diagonal(tmp_array, tmp_array.diagonal() - 1)
			self.variance_parameters += np.dot(np.dot(tmp_array, self.variance_parameters_guess), tmp_array.T)
		else:
			# In that case, the variance of the noise cannot be estimated from empirical data.
			# Estimating Var(X) from guess leads to:
			self.variance_parameters = self.variance_parameters_guess # In other words, there is simply no empirical contribution to the variance of the model parameters.
		
		return(self.variance_parameters)
	
	def scratch_pseudo_inverse_computation(self):
		"""Compute the pseudo-inverse matrix for the model, from scratch.
		
		Parameters:
			self (Model object): Model object.
		
		Returns:
			pseudo_inverse (2D array)
		
		Computational details:
			The full system is rank-decomposed into A = B.C where B is a purely overdetermined system and C is a purely underdetermined system.
			Therefore, the full pseudo-inverse pinv(A) = pinv(B.C) can be written tr(C).inv(C.tr(C)).(tr(B).B).tr(B) = tr(dual(C)).inv(P).tr(B).
		"""
		# Ensures required storage is enabled
		assert(self.decomposed_observations_storage)
		
		# Compute full pseudo-inverse
		# Cost: time: O(nmr+nr²) ; space: O(nr+nm)
		self.pseudo_inverse = np.dot(self.dual_observations_basis.T, np.dot(self.sum_proj_inv, self.decomposed_observations.T))
		
		return(self.pseudo_inverse)
	
	def add_new_regressors(self, nb_new_var=None, new_param_guess=0, new_variances_guess=None):
		"""Add unprecedently seen regressors to the model.
		
		Parameters:
			self (Model object): Model object to update.
			
			nb_new_var (optional, int): Number of new regressors to add.
				Note: if None, the number of new regressors is taken as the first dimension of `new_param_guess`.
				Default: None
			
			new_param_guess (optional, 1D float array or float): Guessed parameters for the new regressors.
				Expecting: dim(new_param_guess) == nb_new_var. The variables will be added in the order provided.
				Note: if None, the new guessed parameters are taken as zero.
				Default: 0
			
			new_variances_guess (optional, 2D float array): Guessed parameters covariance for the new regressors.
				Expecting: dim(new_variances_guess) == nb_new_var. The variables will be added in the order provided.
				Note: if None, the new guessed variances are taken as the average of current guessed variances.
				Default: None
		
		Returns:
			None
		
		Computational details:
			When discovering and introducing new contribution terms, the model must be updated (m increase).
			Such modifications does not impact the decomposed observations (matrix B), if the corresponding contribution terms were not observed before.
			However, the observations basis (matrix C) is slightly impacted:
				- C has additional zero-valued columns
				- dual(C) = inv((C|0).tr((C|0))).(C|0) = inv(C.tr(C))(C|0) has additional zero-valued columns
				- pinv(A) has additional zero-valued lines (if pseudo-inverse update is requested)
		"""
		# Retrieve number of new regressors
		if nb_new_var is None:
			nb_new_var = np.array(np.array(new_param_guess).shape[0])
		
		# Update number of terms
		self.nb_regressors += nb_new_var
		
		# Update core configurations
		tmp_array = np.zeros((self.nb_core_obs, self.nb_regressors), dtype=self.observations_basis.dtype)
		tmp_array[:, :-nb_new_var] = self.observations_basis
		self.observations_basis = tmp_array
		
		tmp_array = np.zeros((self.nb_core_obs, self.nb_regressors), dtype=self.observations_basis.dtype)
		tmp_array[:, :-nb_new_var] = self.dual_observations_basis
		self.dual_observations_basis = tmp_array
		
		# Update regressors parameters
		if new_param_guess is None:
			new_param_guess = 0
		tmp_array = np.empty(self.nb_regressors, dtype=(self.parameters.flat[:0]+np.array(new_param_guess).flat[:0]).dtype) # Hack to get englobing dtype
		tmp_array[:-nb_new_var] = self.parameters
		tmp_array[-nb_new_var:] = new_param_guess
		self.parameters = tmp_array
		
		# Update covariance related storage
		if self.observations_storage:
			# Update raw observations storage
			tmp_array = np.zeros((self.nb_obs, self.nb_regressors), dtype=self.observations.dtype)
			tmp_array[:, :-nb_new_var] = self.observations
			self.observations = tmp_array
			
			# Convert new_variances_guess into array
			if self.empirical_variance:
				if new_variances_guess is not None:
					warnings.warn('Purely empirical variance is enabled, but guessed variance is also provided. Guessed variance will be discarded.')
				new_variances_guess = np.eye(nb_new_var)
			elif new_variances_guess is None:
				new_variances_guess = np.zeros((nb_new_var, nb_new_var))
				np.fill_diagonal(new_variances_guess, self.variance_parameters_guess.diagonal().mean())
			elif np.isscalar(new_variances_guess):
				new_variances_guess = new_variances_guess*np.eye(nb_new_var, dtype=np.array(new_variances_guess).dtype)
			if self.covariance_update: # Covariance update formula assumes symmetric guessed covariance matrix
				try:
					assert(np.allclose(new_variances_guess, new_variances_guess.T))
				except TypeError:
					assert(np.all(new_variances_guess == new_variances_guess.T))
			
			# Update guessed covariance matrix
			# Assume new variables are uncorrelated with previous variables
			tmp_array = np.zeros((self.nb_regressors, self.nb_regressors), dtype=(self.variance_parameters_guess.flat[:0]+new_variances_guess.flat[:0]).dtype) # Hack to get englobing dtype
			tmp_array[:-nb_new_var, :-nb_new_var] = self.variance_parameters_guess
			tmp_array[-nb_new_var:, -nb_new_var:] = new_variances_guess
			self.variance_parameters_guess = tmp_array
		
		# Update variance matrix components
		if self.covariance_update:
			# Update guess variance component
			# Assume new variables are uncorrelated with previous variables
			tmp_array = np.zeros((self.nb_regressors, self.nb_regressors), dtype=(self.guess_variance.flat[:0]+new_variances_guess.flat[:0]).dtype) # Hack to get englobing dtype
			tmp_array[:-nb_new_var, :-nb_new_var] = self.guess_variance
			tmp_array[-nb_new_var:, -nb_new_var:] = new_variances_guess
			self.guess_variance = tmp_array
			
			# Update normalized empiric variance component
			tmp_array = np.zeros((self.nb_regressors, self.nb_regressors), dtype=(self.norm_empiric_variance.flat[:0]+new_variances_guess.flat[:0]).dtype) # Same type as self.guess_variance
			tmp_array[:-nb_new_var, :-nb_new_var] = self.norm_empiric_variance
			self.norm_empiric_variance = tmp_array
			
			# Update combined covariance matrix
			self._update_covariance()
		
		# Update full pseudo-inverse matrix
		if self.pseudo_inverse_update:
			tmp_array = np.zeros((self.nb_regressors, self.nb_obs), dtype=self.pseudo_inverse.dtype)
			tmp_array[:-nb_new_var] = self.pseudo_inverse
			self.pseudo_inverse = tmp_array
		
		return(None)
	
	@property
	def observations_storage(self):
		"""Determine whether full observations must be stored (A matrix and Y vector)"""
		
		# Matrix A and vector Y are required for residuals computation
		return(self.covariance_update or self.covariance_support or self.full_storage)
	
	@property
	def decomposed_observations_storage(self):
		"""Determine whether decomposed observations must be stored (B matrix)"""
		
		# Matrix B is required for pseudo-inverse computation
		return(self.pseudo_inverse_update or self.pseudo_inverse_support or self.full_storage)
	
	@property
	def decomp_obs(self):
		"""Compute decompositon of new observation in the observations basis: γ = dual(C).Γ"""
		
		# Return data if already computed
		if self._decomp_obs is not None:
			return(self._decomp_obs)
		
		# Assert required data are defined
		assert(self._new_observation is not None)
		
		# Decompose new_observation in the core_observations basis (γ = dual(C).Γ)
		# Cost: time: O(mr) ; space: O(r)
		self._decomp_obs = np.dot(self.dual_observations_basis, self._new_observation)
		
		return(self._decomp_obs)
	
	@property
	def obs_reject(self):
		"""Compute rejection vector of new_observation with respect to the observations basis"""
		
		# Return data if already computed
		if self._obs_reject is not None:
			return(self._obs_reject)
		
		# Assert required data are defined
		assert(self._new_observation is not None)
		
		# Compute rejection vector
		if self.nb_core_obs > 0:
			# Get rejection vector of new_observation with respect to the core_observations basis (obs_reject = Γ' = tr(C).γ)
			# Cost: time: O(mr) ; space: O(m)
			self._obs_reject = self._new_observation - np.dot(self.observations_basis.T, self.decomp_obs)
		else:
			# First recording...
			# Cost: time: O(1) ; space: O(1)
			self._obs_reject = self._new_observation
		
		return(self._obs_reject)
	
	@property
	def proj_obs(self):
		"""Compute intermediary result ζ = inv(P).γ = inv(P).dual(C).Γ"""
		
		# Return data if already computed
		if self._proj_obs is not None:
			return(self._proj_obs)
		
		# Compute intermediary result ζ = inv(P).γ
		# Cost: time: O(r²) ; space: O(r)
		self._proj_obs = np.dot(self.sum_proj_inv, self.decomp_obs)
		
		return(self._proj_obs)
	
	@property
	def pinv_obs(self):
		"""Compute intermediary result β = B.ζ = B.inv(P).γ = B.inv(P).dual(C).Γ = tr(pinv(A)).Γ"""
		
		# Return data if already computed
		if self._pinv_obs is not None:
			return(self._pinv_obs)
		
		# Compute intermediary result β = B.ζ
		# Cost: time: O(nr) ; space: O(n)
		assert(self.pseudo_inverse_update)
		self._pinv_obs = np.dot(self.decomposed_observations, self.proj_obs)
		
		return(self._pinv_obs)
	
	@property
	def var_noise(self):
		"""Return variance of the noise for current model."""
		
		# Return infinite variance of the noise if ill-defined
		if not self.nb_obs > self.nb_core_obs:
			return(math.inf)
		
		# Ensures required storage is enabled
		assert(self.observations_storage)
		
		# First, compute err, the error vector
		if self.nb_core_obs > 0:
			# Cost: time: O(nm) ; space: O(n)
			err = self.target_values - np.dot(self.observations, self.parameters)
		else:
			err = self.target_values
		
		# Estimate the variance of the noise: Var_norm(err)
		# Empirical variance of the noise can be computed: var_noise = Var_norm(err) = sum(err**2)/(n-r)
		return(np.dot(err, err) / (self.nb_obs - self.nb_core_obs))
	
	def _purge_temp_storage(self):
		self._new_observation = None
		self._decomp_obs = None
		self._obs_reject = None
		self._proj_obs = None
		self._pinv_obs = None

class RecursiveModelOrthogonal(RecursiveModel):
	"""Orthogonal rank-decomposed linear model with recursive capabilities."""
	###################
	#    Notations    #
	###################
	# X		<=> self.parameters
	# Y		<=> self.target_values
	# A		<=> self.observations
	# B		<=> self.decomposed_observations
	# C		<=> self.observations_basis
	# C̃		<=> self.dual_observations_basis = dual(C) = inv(σ).C = inv(C.tr(C)).C
	# P⁻¹		<=> self.sum_proj_inv = inv(P) = inv(tr(B).B)
	# A⁺		<=> self.pseudo_inverse = pinv(A) = pinv(B.C) = tr(dual(C)).inv(P).tr(B))
	# Γ		<=> self._new_observation (new observation)
	# γ		<=> self.decomp_obs (decompositon of new observation in the observations basis) = dual(C).Γ
	# Γ'		<=> self.obs_reject (rejection vector of new_observation with respect to the observations basis) = Γ - tr(C).γ
	# α		<=> rescaling_factor (rescaling factor for the rejection vector)
	# ζ		<=> self.proj_obs = inv(P).γ = inv(P).dual(C).Γ
	# β		<=> self.pinv_obs = B.ζ = B.inv(P).γ = B.inv(P).dual(C).Γ = tr(pinv(A)).Γ
	# Var(X)	<=> self.variance_parameters
	# nVar(X)_empiric	<=> self.norm_empiric_variance = Var(x)_empiric/Var_norm(err)
	# Var(X)_guess	<=> self.guess_variance
	# Var(g)	<=> self.variance_parameters_guess
	# Var_norm(err) <=> var_noise (empirical estimation of the variance of the noise) = sum(err**2)/(n-r) (= s**2)
	# err		<=> err (residuals) = Y - A.X
	# X'		<=> X' (parameters of the purely overdetermined system, i.e. B.X' = Y) = pinv(B).Y
	# σ		<=> sigma (Gram matrix of observations basis) = C.tr(C)
	# n		= self.nb_obs (number of observations)
	# r		= self.nb_core_obs = |observations basis| = rank of A
	# m		= self.nb_regressors (number of regressors) 
	
	def __init__(self, eps=None, pseudo_inverse_update=False, pseudo_inverse_support=False, covariance_support=False, full_storage=False, empirical_variance=False):
		super(self.__class__, self).__init__(eps=eps, pseudo_inverse_update=pseudo_inverse_update, covariance_update=False, pseudo_inverse_support=pseudo_inverse_support, covariance_support=covariance_support, full_storage=full_storage, empirical_variance=empirical_variance)
	
	#################################
	#    Formulae implementation    #
	#################################
	
	#
	# Linear algebra details can be found on ...
	#
	# Here, we are considering an orthogonal basis (non-rescaled) rank factorisation updating scheme.
	# As described in the paper above, the only changes occur in the update of internal storages when adding a linearly independant row
	
	def _core_update_model_storage(self, kalman_gain, new_value, new_observation=None, obs_reject=None, decomp_obs=None, proj_obs=None):
		"""Update linear model parameters by adding a new configuration that is linearly independant with previous configurations.
		
		Parameters:
			self (Model object): Model object.
			
			kalman_gain (1D array): Associated Kalman gain.
			
			new_value (float): Associated target value for the current new observation.
		"""
		
		if new_observation is None:
			new_observation = self._new_observation
		
		if obs_reject is None:
			# Use internally computed Γ' vector
			# Cost: None (already computed)
			obs_reject = self.obs_reject
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		if proj_obs is None:
			# Use internally computed ζ vector
			# Cost: time: O(r²) ; space: O(r)
			proj_obs = self.proj_obs
		
		# Define rescaling factor: α
		# Cost: time: O(1) ; space: O(1)
		rescaling_factor = 1
		inv_rescl_factor = 1 # = 1/rescaling_factor
		
		# Update observation basis: C_new = (C/tr(αΓ'))
		# Cost: time: O(mr) ; space: O(mr)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_regressors), order='F', dtype=kalman_gain.dtype)
		tmp_array[:-1] = self.observations_basis
		tmp_array[-1] = rescaling_factor*obs_reject
		self.observations_basis = tmp_array
		
		# Update dual of observations basis: dual(C_new) = (dual(C)/tr(kalman_gain/α))
		# Cost: time: O(1) ; space: O(1)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_regressors), dtype=kalman_gain.dtype)
		tmp_array[:-1] = self.dual_observations_basis
		tmp_array[-1] = inv_rescl_factor*kalman_gain
		self.dual_observations_basis = tmp_array
		
		# Update inv(P) matrix: inv(P_new) = (inv(P) / -tr(αζ) | -αζ / α²(1+tr(γ).ζ)
		# Cost: time: O(r²) ; space: O(r²)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
		tmp_array[:-1, :-1] = self.sum_proj_inv
		tmp_array[-1, :-1] = tmp_array[:-1, -1] = -rescaling_factor*proj_obs
		tmp_array[-1, -1] = rescaling_factor**2*(1 + (np.dot(decomp_obs, proj_obs) or 0)) # default value 0 if tr(γ).ζ is None
		self.sum_proj_inv = tmp_array
		
		if self.decomposed_observations_storage:
			# Update decomposed observations: B_new = (B / tr(γ) | 0 / inv(a))
			# Cost: time: O(nr) ; space: O(nr)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
			tmp_array[:-1, :-1] = self.decomposed_observations
			tmp_array[:-1, -1] = 0
			tmp_array[-1, :-1] = decomp_obs
			tmp_array[-1, -1] = inv_rescl_factor
			self.decomposed_observations = tmp_array
		
		if self.observations_storage:
			# Update raw observations: A_new = (A/tr(Γ))
			# Cost: time: O(mn) ; space: O(mn)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_regressors), dtype=kalman_gain.dtype)
			tmp_array[:-1] = self.observations
			tmp_array[-1] = new_observation
			self.observations = tmp_array
			
			# Update regressand target values:
			self.target_values = np.append(self.target_values, new_value)
		
		# Update internal counters
		self.nb_core_obs += 1
		self.nb_obs += 1

class RecursiveModelOrthonormal(RecursiveModel):
	"""Orthonormal rank-decomposed linear model with recursive capabilities."""
	###################
	#    Notations    #
	###################
	# X		<=> self.parameters
	# Y		<=> self.target_values
	# A		<=> self.observations
	# B		<=> self.decomposed_observations
	# C		<=> self.observations_basis
	# C̃		<=> self.dual_observations_basis = dual(C) = inv(σ).C = inv(C.tr(C)).C
	# P⁻¹		<=> self.sum_proj_inv = inv(P) = inv(tr(B).B)
	# A⁺		<=> self.pseudo_inverse = pinv(A) = pinv(B.C) = tr(dual(C)).inv(P).tr(B))
	# Γ		<=> self._new_observation (new observation)
	# γ		<=> self.decomp_obs (decompositon of new observation in the observations basis) = dual(C).Γ
	# Γ'		<=> self.obs_reject (rejection vector of new_observation with respect to the observations basis) = Γ - tr(C).γ
	# α		<=> rescaling_factor (for orthonormal storage: inverse of rejection vector norm) = 1/||Γ'||
	# ζ		<=> self.proj_obs = inv(P).γ = inv(P).dual(C).Γ
	# β		<=> self.pinv_obs = B.ζ = B.inv(P).γ = B.inv(P).dual(C).Γ = tr(pinv(A)).Γ
	# Var(X)	<=> self.variance_parameters
	# nVar(X)_empiric	<=> self.norm_empiric_variance = Var(x)_empiric/Var_norm(err)
	# Var(X)_guess	<=> self.guess_variance
	# Var(g)	<=> self.variance_parameters_guess
	# Var_norm(err) <=> var_noise (empirical estimation of the variance of the noise) = sum(err**2)/(n-r) (= s**2)
	# err		<=> err (residuals) = Y - A.X
	# X'		<=> X' (parameters of the purely overdetermined system, i.e. B.X' = Y) = pinv(B).Y
	# σ		<=> sigma (Gram matrix of observations basis) = C.tr(C)
	# n		= self.nb_obs (number of observations)
	# r		= self.nb_core_obs = |observations basis| = rank of A
	# m		= self.nb_regressors (number of regressors) 
	
	def __init__(self, eps=None, pseudo_inverse_update=False, pseudo_inverse_support=False, covariance_support=False, full_storage=False, empirical_variance=False):
		super(self.__class__, self).__init__(eps=eps, pseudo_inverse_update=pseudo_inverse_update, covariance_update=False, pseudo_inverse_support=pseudo_inverse_support, covariance_support=covariance_support, full_storage=full_storage, empirical_variance=empirical_variance)
	
	#################################
	#    Formulae implementation    #
	#################################
	
	#
	# Linear algebra details can be found on ...
	#
	# Here, we are considering an orthonormal basis rank factorisation updating scheme.
	# As described in the paper above, the only changes occur in the update of internal storages when adding a linearly independant row
	
	def _core_update_model_storage(self, kalman_gain, new_value, new_observation=None, obs_reject=None, decomp_obs=None, proj_obs=None):
		"""Update linear model parameters by adding a new configuration that is linearly independant with previous configurations.
		
		Parameters:
			self (Model object): Model object.
			
			kalman_gain (1D array): Associated Kalman gain.
			
			new_value (float): Associated target value for the current new observation.
		"""
		
		if new_observation is None:
			new_observation = self._new_observation
		
		if obs_reject is None:
			# Use internally computed Γ' vector
			# Cost: None (already computed)
			obs_reject = self.obs_reject
		
		if decomp_obs is None:
			# Use internally computed γ vector
			# Cost: None (already computed)
			decomp_obs = self.decomp_obs
		
		if proj_obs is None:
			# Use internally computed ζ vector
			# Cost: time: O(r²) ; space: O(r)
			proj_obs = self.proj_obs
		
		# Compute rescaling factor: α = 1/||Γ'||
		# Cost: time: O(m) ; space: O(1)
		obs_rej_norm = np.linalg.norm(self.obs_reject)
		rescaling_factor = 1/obs_rej_norm
		
		# Update observation basis: C_new = (C/tr(αΓ'))
		# Cost: time: O(mr) ; space: O(mr)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_regressors), order='F', dtype=kalman_gain.dtype)
		tmp_array[:-1] = self.observations_basis
		tmp_array[-1] = rescaling_factor*obs_reject
		self.observations_basis = tmp_array
		
		# Update dual of observations basis: dual(C_new) = C_new
		# Cost: time: O(1) ; space: O(1)
		self.dual_observations_basis = self.observations_basis
		
		# Update inv(P) matrix: inv(P_new) = (inv(P) / -tr(αζ) | -αζ / α²(1+tr(γ).ζ)
		# Cost: time: O(r²) ; space: O(r²)
		tmp_array = np.empty((self.nb_core_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
		tmp_array[:-1, :-1] = self.sum_proj_inv
		tmp_array[-1, :-1] = tmp_array[:-1, -1] = -rescaling_factor*proj_obs
		tmp_array[-1, -1] = rescaling_factor**2*(1 + (np.dot(decomp_obs, proj_obs) or 0)) # default value 0 if tr(γ).ζ is None
		self.sum_proj_inv = tmp_array
		
		if self.decomposed_observations_storage:
			# Update decomposed observations: B_new = (B / tr(γ) | 0 / inv(a))
			# Cost: time: O(nr) ; space: O(nr)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_core_obs + 1), dtype=kalman_gain.dtype)
			tmp_array[:-1, :-1] = self.decomposed_observations
			tmp_array[:-1, -1] = 0
			tmp_array[-1, :-1] = decomp_obs
			tmp_array[-1, -1] = obs_rej_norm
			self.decomposed_observations = tmp_array
		
		if self.observations_storage:
			# Update raw observations: A_new = (A/tr(Γ))
			# Cost: time: O(mn) ; space: O(mn)
			tmp_array = np.empty((self.nb_obs + 1, self.nb_regressors), dtype=kalman_gain.dtype)
			tmp_array[:-1] = self.observations
			tmp_array[-1] = new_observation
			self.observations = tmp_array
			
			# Update regressand target values:
			self.target_values = np.append(self.target_values, new_value)
		
		# Update internal counters
		self.nb_core_obs += 1
		self.nb_obs += 1
