#!/usr/bin/env python3

import unittest
import numpy as np
import scipy.linalg as scla
from fractions import Fraction
from context import rank_greville

### Tools

def generate_normal_dist_matrix(n, m, r):
	"""Returns a Normal N(0,1) distributed random matrix with size n x m of rank <= r"""
	np.random.seed(0)
	coefs = 1-2*np.random.rand(n, r) # Ensures mean == 0
	coefs /= np.sqrt((coefs**2).sum(axis=-1, keepdims=True)) # Sum of squared coefficients is 1 for each observation (ensures std == 1)
	
	return(np.array(np.dot(coefs, np.random.randn(r, m))))

### Tests

class ArrayTestCase(unittest.TestCase):
	def assertArrayEqual(self, array1, array2):
		np.testing.assert_array_equal(array1, array2)
	
	def assertArrayAlmostEqual(self, array1, array2):
		np.testing.assert_array_almost_equal(array1, array2)

def model_generator(model_class):
	class ModelGenerator(object):
		def __init__(self, **kwargs):
			self.model = None
			self.observations = None
			self.guess = None
			self.covariance_guess = None
			self.target_values = None
			self.kwargs = kwargs
	
		def compute_covariance_projectors(self, lstsq, pinv):
			res = self.values - np.dot(self.observations, lstsq)
			rank = np.linalg.matrix_rank(self.observations)
			var_noise = np.dot(res, res) / (len(self.observations) - rank)
			self.proj_cov = np.dot(np.dot(pinv, np.eye(len(self.values))*var_noise), pinv.T)
			null_space = np.linalg.svd(self.observations)[2][rank:]
			self.P_ker_A = np.dot(null_space.T, null_space)
			self.P_im_trA = np.eye(len(self.guess)) - self.P_ker_A
	
		def gen_single(self):
			self.observations = [[1, 1]]
			self.values = [2]
			self.model = model_class(**self.kwargs)
			return(self.model.add_observation(self.observations[0], self.values[0]))
	
		def gen_identity(self):
			self.observations = [[1, 0],
				             [0, 1]]
			self.values = [2, 3]
			self.covariance_guess = np.random.rand(2, 2)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(2, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_overdetermined(self):
			self.observations = [[1, 1],
				             [1, 0],
				             [0, 1]]
			self.values = [6, 2, 3]
			self.model = model_class(**self.kwargs)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_overdetermined_exact(self):
			self.observations = [[1, 1],
				             [1, 0],
				             [0, 1]]
			self.values = [4, 2, 2]
			self.model = model_class(**self.kwargs)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_underdetermined(self):
			self.observations = [[1, 1, 1],
				             [1, 0, 2]]
			self.values = [6, -2]
			self.model = model_class(**self.kwargs)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_underdetermined_guess(self):
			self.observations = [[1, 1, 1],
				             [1, 0, 2]]
			self.values = [6, -2]
			self.guess = [14, 0, -8]
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_over_underdetermined(self):
			self.observations = [[1, 1, 1],
				             [1, 1, 0],
				             [0, 0, 1]]
			self.values = [6, 2, 3]
			self.model = model_class(**self.kwargs)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_over_underdetermined_guess(self):
			self.observations = [[1, 1, 1],
				             [1, 1, 0],
				             [0, 0, 1]]
			self.values = [6, 2, 3]
			self.guess = [10, 0, 0]
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_random_square(self):
			np.random.seed(0)
			self.observations = generate_normal_dist_matrix(5, 5, 5)
			self.values = np.random.randn(5)
			eps = 1e-8
			self.model = model_class(**self.kwargs, eps=eps)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_random_overdetermined(self):
			np.random.seed(0)
			self.observations = generate_normal_dist_matrix(5, 3, 3)
			self.values = np.random.randn(5)
			eps = 1e-8
			self.model = model_class(**self.kwargs, eps=eps)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_random_underdetermined(self):
			np.random.seed(0)
			self.observations = generate_normal_dist_matrix(3, 5, 3)
			self.values = np.random.randn(3)
			eps = 1e-8
			self.model = model_class(**self.kwargs, eps=eps)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_random_over_underdetermined(self):
			np.random.seed(0)
			self.observations = generate_normal_dist_matrix(4, 5, 3)
			self.values = np.random.randn(4)
			eps = 1e-8
			self.model = model_class(**self.kwargs, eps=eps)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_random_over_underdetermined_cov_guess(self):
			np.random.seed(0)
			self.observations = generate_normal_dist_matrix(5, 4, 3)
			self.values = np.random.randn(5)
			self.guess = np.random.randn(4)
			self.covariance_guess = np.random.rand(4, 4)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			eps = 1e-8
			self.model = model_class(**self.kwargs, eps=eps)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_null_obs(self):
			self.observations = [[1, 1, 1],
				             [0, 0, 0],
				             [1, 1, 0],
				             [0, 0, 1]]
			self.values = [6, 1, 2, 3]
			self.guess = [10, 0, 0]
			self.covariance_guess = np.random.rand(3, 3)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_first_null(self):
			self.observations = [[0, 0, 0],
				             [1, 1, 1],
				             [1, 1, 0],
				             [0, 0, 1]]
			self.values = [1, 6, 2, 3]
			self.guess = [10, 0, 0]
			self.covariance_guess = np.random.rand(3, 3)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_single_null(self):
			self.observations = [[0, 0, 0]]
			self.values = [-1]
			self.guess = [10, 0, 0]
			self.covariance_guess = np.random.rand(3, 3)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_pascal(self):
			self.observations = scla.pascal(3)
			self.values = [0]*3
			self.model = model_class(**self.kwargs)
			return(self.model.add_observations(self.observations, self.values))
		
		def gen_pascal_fractions(self):
			self.observations = scla.pascal(3).astype(int) + Fraction()
			self.values = [Fraction(1)]*3
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(nb_new_var=3, new_param_guess=Fraction(0), new_variances_guess=Fraction(1))
			return(self.model.add_observations(self.observations, self.values))
		
		def gen_pascal_overdetermined_fractions(self):
			self.observations = scla.pascal(3).astype(int) + Fraction()
			self.observations = np.vstack((self.observations, [1, 0, 0]))
			self.values = [Fraction(1)]*3+[2]
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(nb_new_var=3, new_param_guess=Fraction(0), new_variances_guess=Fraction(1))
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_overdetermined_identity_dual(self):
			self.observations = [[1, 0],
				             [0, 1],
				             [1, 1]]
			self.values = [2, 3, 8]
			self.guess = [0, 0]
			self.covariance_guess = np.random.rand(2, 2)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_overdetermined_diagonal_cov_guess(self):
			self.observations = [[2, 0],
				             [0, 1],
				             [1, 1]]
			self.values = [2, 3, 8]
			self.guess = [0, 0]
			self.covariance_guess = np.eye(2) * [2, 1]
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	
		def gen_overdetermined_non_trivial_dual(self):
			self.observations = [[1, 1],
				             [0, 1],
				             [1, 0]]
			self.values = [8, 3, 2]
			self.guess = [0, 0]
			self.covariance_guess = np.random.rand(2, 2)
			self.covariance_guess = (self.covariance_guess + self.covariance_guess.T)/2
			self.model = model_class(**self.kwargs)
			self.model.add_new_regressors(new_param_guess=self.guess, new_variances_guess=self.covariance_guess)
			return(self.model.add_observations(self.observations, self.values))
	return(ModelGenerator)

def test_least_squares(ModelGenerator):
	class TestLeastSquares(ArrayTestCase):
		"""Test least-squares correctness using default config"""
		def test_single(self):
			gen = ModelGenerator()
			update = gen.gen_single()
			self.assertArrayEqual(update, [1, 1])
			self.assertArrayEqual(gen.model.parameters, [1, 1])
	
		def test_identity(self):
			gen = ModelGenerator()
			update = gen.gen_identity()
			self.assertArrayEqual(update, [2, 3])
			self.assertArrayEqual(gen.model.parameters, [2, 3])
	
		def test_overdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_overdetermined()
			self.assertArrayAlmostEqual(update, [2+1/3, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [2+1/3, 3+1/3])
	
		def test_overdetermined_exact(self):
			gen = ModelGenerator()
			update = gen.gen_overdetermined_exact()
			self.assertArrayEqual(update, [2, 2])
			self.assertArrayEqual(gen.model.parameters, [2, 2])
	
		def test_underdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_underdetermined()
			self.assertArrayAlmostEqual(update, [2, 6, -2])
			self.assertArrayAlmostEqual(gen.model.parameters, [2, 6, -2])
	
		def test_underdetermined_guess(self):
			gen = ModelGenerator()
			update = gen.gen_underdetermined_guess()
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [14, 0, -8])
	
		def test_over_underdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_over_underdetermined()
			self.assertArrayAlmostEqual(update, [1+1/6, 1+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [1+1/6, 1+1/6, 3+1/3])
	
		def test_over_underdetermined_guess(self):
			gen = ModelGenerator()
			update = gen.gen_over_underdetermined_guess()
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_random_square(self):
			gen = ModelGenerator()
			update = gen.gen_random_square()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_overdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_random_overdetermined()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_underdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_random_underdetermined()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined(self):
			gen = ModelGenerator()
			update = gen.gen_random_over_underdetermined()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess(self):
			gen = ModelGenerator()
			update = gen.gen_random_over_underdetermined_cov_guess()
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs(self):
			gen = ModelGenerator()
			update = gen.gen_null_obs()
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_first_null(self):
			gen = ModelGenerator()
			update = gen.gen_first_null()
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_single_null(self):
			gen = ModelGenerator()
			update = gen.gen_single_null()
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	return(TestLeastSquares)

def test_pseudoinverse(ModelGenerator):
	class TestPseudoinverse(ArrayTestCase):
		"""Test pseudoinverse correctness"""
		def test_identity(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			update = gen.gen_identity()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.eye(2)
			self.assertArrayEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayEqual(update, [2, 3])
			self.assertArrayEqual(gen.model.parameters, [2, 3])
	
		def test_random_square(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_random_square()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_overdetermined(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_random_overdetermined()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_underdetermined(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_random_underdetermined()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_random_over_underdetermined()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_random_over_underdetermined_cov_guess()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			update = gen.gen_null_obs()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_first_null(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			update = gen.gen_first_null()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_single_null(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			update = gen.gen_single_null()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	
		def test_pascal(self):
			gen = ModelGenerator(pseudo_inverse_support=True)
			gen.gen_pascal()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			pinv = np.linalg.inv(gen.observations).round().astype(int)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
	
		def test_identity_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			update = gen.gen_identity()
			pinv = np.eye(2)
			self.assertArrayEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayEqual(update, [2, 3])
			self.assertArrayEqual(gen.model.parameters, [2, 3])
	
		def test_random_square_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_random_square()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_overdetermined_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_random_overdetermined()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_underdetermined_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_random_underdetermined()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_random_over_underdetermined()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_random_over_underdetermined_cov_guess()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			update = gen.gen_null_obs()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_first_null_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			update = gen.gen_first_null()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, [10-4+1/6, -4+1/6, 3+1/3])
	
		def test_single_null_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			update = gen.gen_single_null()
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	
		def test_pascal_update(self):
			gen = ModelGenerator(pseudo_inverse_update=True)
			gen.gen_pascal()
			pinv = np.linalg.inv(gen.observations).round().astype(int)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
	return(TestPseudoinverse)

def test_covariance(ModelGenerator):
	class TestCovariance(ArrayTestCase):
		"""Test covariance correctness"""
		def test_identity(self):
			gen = ModelGenerator(covariance_support=True)
			update = gen.gen_identity()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			lstsq = [2, 3]
			self.assertArrayEqual(update, lstsq)
			self.assertArrayEqual(gen.model.parameters, lstsq)
	
	#	def test_random_square(self):
	#		gen = ModelGenerator(covariance_support=True)
	#		gen.gen_random_square()
	#		variance_parameters = gen.model.scratch_covariance_computation()
	#		self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
	#		cov = gen.covariance_guess
	#		self.assertArrayEqual(gen.model.variance_parameters, cov)
	#		lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
	#		self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_identity_dual(self):
			gen = ModelGenerator(covariance_support=True)
			gen.gen_overdetermined_identity_dual()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_diagonal_cov_guess(self):
			gen = ModelGenerator(covariance_support=True)
			gen.gen_overdetermined_diagonal_cov_guess()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_non_trivial_dual(self):
			gen = ModelGenerator(covariance_support=True)
			gen.gen_overdetermined_non_trivial_dual()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess(self):
			gen = ModelGenerator(covariance_support=True)
			gen.gen_random_over_underdetermined_cov_guess()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs(self):
			gen = ModelGenerator(covariance_support=True)
			update = gen.gen_null_obs()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_first_null(self):
			gen = ModelGenerator(covariance_support=True)
			update = gen.gen_first_null()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_single_null(self):
			gen = ModelGenerator(covariance_support=True)
			update = gen.gen_single_null()
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	
		def test_identity_update(self):
			gen = ModelGenerator(covariance_update=True)
			update = gen.gen_identity()
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			lstsq = [2, 3]
			self.assertArrayEqual(update, lstsq)
			self.assertArrayEqual(gen.model.parameters, lstsq)
	
	#	def test_random_square_update(self):
	#		gen = ModelGenerator(covariance_update=True)
	#		gen.gen_random_square()
	#		cov = gen.covariance_guess
	#		self.assertArrayEqual(gen.model.variance_parameters, cov)
	#		lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
	#		self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_identity_dual_update(self):
			gen = ModelGenerator(covariance_update=True)
			gen.gen_overdetermined_identity_dual()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_diagonal_cov_guess_update(self):
			gen = ModelGenerator(covariance_update=True)
			gen.gen_overdetermined_diagonal_cov_guess()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_non_trivial_dual_update(self):
			gen = ModelGenerator(covariance_update=True)
			gen.gen_overdetermined_non_trivial_dual()
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess_update(self):
			gen = ModelGenerator(covariance_update=True)
			gen.gen_random_over_underdetermined_cov_guess()
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs_update(self):
			gen = ModelGenerator(covariance_update=True)
			update = gen.gen_null_obs()
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_first_null_update(self):
			gen = ModelGenerator(covariance_update=True)
			update = gen.gen_first_null()
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_single_null_update(self):
			gen = ModelGenerator(covariance_update=True)
			update = gen.gen_single_null()
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	return(TestCovariance)

def test_full_storage(ModelGenerator):
	class TestFullStorage(ArrayTestCase):
		"""Test full storage features"""
		def test_identity(self):
			gen = ModelGenerator(full_storage=True)
			update = gen.gen_identity()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			pinv = np.eye(2)
			self.assertArrayEqual(gen.model.pseudo_inverse, pinv)
			lstsq = [2, 3]
			self.assertArrayEqual(update, lstsq)
			self.assertArrayEqual(gen.model.parameters, lstsq)
	
	#	def test_random_square(self):
	#		gen = ModelGenerator(full_storage=True)
	#		gen.gen_random_square()
	#		pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
	#		self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
	#		variance_parameters = gen.model.scratch_covariance_computation()
	#		self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
	#		cov = gen.covariance_guess
	#		self.assertArrayEqual(gen.model.variance_parameters, cov)
	#		pinv = np.linalg.pinv(gen.observations)
	#		self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
	#		lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
	#		self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_identity_dual(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_overdetermined_identity_dual()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_diagonal_cov_guess(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_overdetermined_diagonal_cov_guess()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_overdetermined_non_trivial_dual(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_overdetermined_non_trivial_dual()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values, rcond=gen.model.eps)[0]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_random_over_underdetermined_cov_guess(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_random_over_underdetermined_cov_guess()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = np.linalg.lstsq(gen.observations, gen.values-np.dot(gen.observations, gen.guess), rcond=gen.model.eps)[0] + gen.guess
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_null_obs(self):
			gen = ModelGenerator(full_storage=True)
			update = gen.gen_null_obs()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_first_null(self):
			gen = ModelGenerator(full_storage=True)
			update = gen.gen_first_null()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			lstsq = [10-4+1/6, -4+1/6, 3+1/3]
			pinv = np.linalg.pinv(gen.observations)
			gen.compute_covariance_projectors(lstsq, pinv)
			self.assertArrayAlmostEqual(np.dot(np.dot(gen.P_im_trA, gen.model.variance_parameters), gen.P_im_trA.T), gen.proj_cov)
			self.assertArrayAlmostEqual(gen.model.variance_parameters, gen.proj_cov + np.dot(np.dot(gen.P_ker_A, gen.covariance_guess), gen.P_ker_A.T))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayAlmostEqual(update, [-4+1/6, -4+1/6, 3+1/3])
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
	
		def test_single_null(self):
			gen = ModelGenerator(full_storage=True)
			update = gen.gen_single_null()
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			cov = gen.covariance_guess
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			pinv = np.linalg.pinv(gen.observations)
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
			self.assertArrayEqual(update, [0, 0, 0])
			self.assertArrayEqual(gen.model.parameters, [10, 0, 0])
	return(TestFullStorage)

def test_fractions(ModelGenerator):
	class TestFractions(ArrayTestCase):
		def test_pascal_fractions(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_pascal_fractions()
			lstsq = scla.lstsq(gen.observations.astype(int), np.array(gen.values).astype(int))[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			cov = np.eye(3).astype(int)+Fraction()
			self.assertArrayEqual(gen.model.variance_parameters, cov)
			pinv = scla.pinv(gen.observations.astype(int)).round().astype(int) + Fraction()
			self.assertArrayEqual(gen.model.pseudo_inverse, pinv)
		
		def test_pascal_overdetermined_fractions(self):
			gen = ModelGenerator(full_storage=True)
			gen.gen_pascal_overdetermined_fractions()
			lstsq = scla.lstsq(gen.observations.astype(int), np.array(gen.values).astype(int))[0]
			self.assertArrayAlmostEqual(gen.model.parameters, lstsq)
			pseudo_inverse = gen.model.scratch_pseudo_inverse_computation()
			self.assertArrayAlmostEqual(pseudo_inverse, gen.model.pseudo_inverse)
			variance_parameters = gen.model.scratch_covariance_computation()
			self.assertArrayEqual(variance_parameters, gen.model.variance_parameters)
			pinv = scla.pinv(gen.observations.astype(int))
			self.assertArrayAlmostEqual(gen.model.pseudo_inverse, pinv)
	return(TestFractions)

# Testing RecursiveModel
class TestLeastSquaresGeneral(test_least_squares(model_generator(rank_greville.RecursiveModel))):
	pass
class TestPseudoinverseGeneral(test_pseudoinverse(model_generator(rank_greville.RecursiveModel))):
	pass
class TestCovarianceGeneral(test_covariance(model_generator(rank_greville.RecursiveModel))):
	pass
class TestFullStorageGeneral(test_full_storage(model_generator(rank_greville.RecursiveModel))):
	pass
class TestFractionsGeneral(test_fractions(model_generator(rank_greville.RecursiveModel))):
	pass

# Testing RecursiveModelOrthogonal
class TestLeastSquaresOrthogonal(test_least_squares(model_generator(rank_greville.RecursiveModelOrthogonal))):
	pass
class TestPseudoinverseOrthogonal(test_pseudoinverse(model_generator(rank_greville.RecursiveModelOrthogonal))):
	pass
class TestFullStorageOrthogonal(test_full_storage(model_generator(rank_greville.RecursiveModelOrthogonal))):
	pass
class TestFractionsOrthogonal(test_fractions(model_generator(rank_greville.RecursiveModelOrthogonal))):
	pass

# Testing RecursiveModelOrthonormal
class TestLeastSquaresOrthonormal(test_least_squares(model_generator(rank_greville.RecursiveModelOrthonormal))):
	pass
class TestPseudoinverseOrthonormal(test_pseudoinverse(model_generator(rank_greville.RecursiveModelOrthonormal))):
	pass
class TestFullStorageOrthonormal(test_full_storage(model_generator(rank_greville.RecursiveModelOrthonormal))):
	pass

if __name__ == '__main__':
	unittest.main()
