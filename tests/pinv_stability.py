#!/usr/bin/env python3

from collections import OrderedDict
import numpy as np
import scipy.linalg as scla
from scipy.stats import ortho_group
import tabulate
import re
from context import sample as rank_greville

def reset_randn(n, m):
	np.random.seed(0)
	return(np.random.randn(n, m))

def reset_rando(n, m=None):
	if m is None:
		m = n
	ortho_group.random_state = 0
	return(ortho_group.rvs(n)[:, :m])

class PinvStabilityTester(object):
	def __init__(self, matrices, pinv_ref, norm_mat=None, norm_pinv_ref=None, cond_nb_mat=None, rcond=None, **kwargs):
		self.kwargs = kwargs
		self.matrices = matrices
		self.pinv_ref = pinv_ref
		self.norm_mat = [scla.svd(matrix)[1].max() for matrix in self.matrices] if norm_mat is None else norm_mat
		self.norm_pinv_ref = [scla.svd(pinv)[1].max() for pinv in self.pinv_ref] if norm_pinv_ref is None else norm_pinv_ref
		self.cond_nb_mat = [scla.svd(matrix)[1].max()/scla.svd(matrix)[1].min() for matrix in self.matrices] if cond_nb_mat is None else cond_nb_mat
		self.eps = np.finfo(np.float).eps
		self.rcond = rcond
	
	def compute_pinv(self):
		self.pinv = OrderedDict()
		
		self.pinv['orthonormal'] = []
		for matrix in self.matrices:
			model = rank_greville.RecursiveModelOrthonormal(pseudo_inverse_update=True, eps=self.rcond, **self.kwargs)
			model.add_observations(matrix, [0]*len(matrix))
			self.pinv['orthonormal'].append(model.pseudo_inverse)
		
		self.pinv['orthogonal'] = []
		for matrix in self.matrices:
			model = rank_greville.RecursiveModelOrthogonal(pseudo_inverse_update=True, eps=self.rcond, **self.kwargs)
			model.add_observations(matrix, [0]*len(matrix))
			self.pinv['orthogonal'].append(model.pseudo_inverse)
		
		self.pinv['rank-Greville'] = []
		for matrix in self.matrices:
			model = rank_greville.RecursiveModel(pseudo_inverse_update=True, eps=self.rcond, **self.kwargs)
			model.add_observations(matrix, [0]*len(matrix))
			self.pinv['rank-Greville'].append(model.pseudo_inverse)
		
		self.pinv['Greville'] = []
		for matrix in self.matrices:
			A = [matrix[0]]
			pinvA = np.array(matrix[0]/np.dot(matrix[0], matrix[0]))[:, None]
			for a in matrix[1:]:
				d = np.dot(a, pinvA)
				c = a - np.dot(d, A)
				m, n = pinvA.shape
				if np.linalg.norm(c) < ((m**2*n+m*n+m)*self.eps if self.rcond is None else self.rcond)*max(1, np.linalg.norm(a)) :
					b = np.dot(pinvA, d)/(1+np.dot(d,d))
				else:
					b = c/np.dot(c,c)
				pinvA = np.hstack((pinvA - np.outer(b, d), b[:,None]))
				A.append(a)
			self.pinv['Greville'].append(pinvA)
		
		self.pinv['Cholesky'] = []
		for matrix in self.matrices:
			try:
				self.pinv['Cholesky'].append(scla.cho_solve(scla.cho_factor(np.dot(matrix.T, matrix)), matrix.T))
			except (scla.LinAlgError, ValueError):
				self.pinv['Cholesky'].append(None)
		
		for algo in 'gelsy', 'gelsd', 'gelss':
			self.pinv[algo] = [scla.lstsq(matrix, np.eye(matrix.shape[0]), cond=self.rcond, lapack_driver=algo)[0] for matrix in self.matrices]
	
	def compute_stability_factor(self):
		self.stability_factor = OrderedDict()
		for algo in self.pinv:
			self.stability_factor[algo] = [scla.svd(pinv.astype(np.float)-pinv_ref)[1].max()/(self.eps*norm_pinv_ref*cond_nb_mat) if pinv is not None else None for pinv, pinv_ref, norm_pinv_ref, cond_nb_mat in zip(self.pinv[algo], self.pinv_ref, self.norm_pinv_ref, self.cond_nb_mat)]
	
	def compute_residual_error(self):
		self.residual_error = OrderedDict()
		for algo in self.pinv:
			self.residual_error[algo] = [scla.svd(np.dot(pinv.astype(np.float), matrix)-np.eye(pinv.shape[0]))[1].max()/(norm_mat*scla.svd(pinv.astype(np.float))[1].max()) if pinv is not None else None for pinv, matrix, norm_mat in zip(self.pinv[algo], self.matrices, self.norm_mat)]
	
	def compute_all(self, **kwargs):
		self.compute_pinv()
		self.compute_stability_factor()
		self.compute_residual_error()
	
	def print_all(self, parameters=OrderedDict({'param':[]}), latex=False, label=None, caption=None):
		tablefmt = 'latex_raw' if latex else 'psql'
		cond_nb_label = r'$\kappa_2(A)$' if latex else 'cond_nb'
		
		if not latex: print('Stability factor:')
		data = parameters.copy()
		if latex:
			data[cond_nb_label] = [re.sub('e([-+])(0*)([0-9]+)', r'\mathrm{e}{\1\3}', r'${:.2e}$'.format(cond_nb)) for cond_nb in self.cond_nb_mat]
			for algo in self.stability_factor:
				algo_label = r'$e_\text{{{:s}}}$'.format(algo)
				data[algo_label] = [re.sub('e([-+])(0*)([0-9]+)', r'\mathrm{e}{\1\3}', r'${:.2e}$'.format(value)) if value is not None else 'failure' for value in self.stability_factor[algo]]
		else:
			data[cond_nb_label] = self.cond_nb_mat
			data.update(self.stability_factor)
		if latex: print(r'\begin{{table}}\
		                  \caption{{\label{{table_e_{}}}Empiric stability factors associated with the pseudoinverse computation of {}.}}\
		                  \makebox[\linewidth]{{'.format(label, caption))
		print(tabulate.tabulate(data,
		                        headers=data,
		                        tablefmt=tablefmt,
		                        floatfmt='.2e',
		                        stralign='left',
		                        numalign='right'))
		if latex: print(r'}\end{table}')
		
		if not latex: print('Residual error:')
		data = parameters.copy()

		if latex:
			data[cond_nb_label] = [re.sub('e([-+])(0*)([0-9]+)', r'\mathrm{e}{\1\3}', r'${:.2e}$'.format(cond_nb)) for cond_nb in self.cond_nb_mat]
			for algo in self.residual_error:
				algo_label = r'$res_\text{{{:s}}}$'.format(algo)
				data[algo_label] = [re.sub('e([-+])(0*)([0-9]+)', r'\mathrm{e}{\1\3}', r'${:.2e}$'.format(value)) if value is not None else 'failure' for value in self.residual_error[algo]]
		else:
			data[cond_nb_label] = self.cond_nb_mat
			data.update(self.residual_error)
		if latex: print(r'\begin{{table}}\
		                  \caption{{\label{{table_res_{}}}Empiric residual errors associated with the pseudoinverse computation of {}.}}\
		                  \makebox[\linewidth]{{'.format(label, caption))
		print(tabulate.tabulate(data,
		                        headers=data,
		                        tablefmt=tablefmt,
		                        floatfmt='.2e',
		                        stralign='left',
		                        numalign='right'))
		if latex: print(r'}\end{table}')
		print('\n')

if __name__ == '__main__':
	LATEX_DISPLAY = False
	tests = OrderedDict()
	eps = np.finfo(np.float).eps
	rcond = None
	
	# Pascal matrices
	if not LATEX_DISPLAY: print('Treating Pascal matrices:')
	n_values = (4, 6, 8, 10)
	matrices = [scla.pascal(n).astype(int) for n in n_values]
	pinv_ref = [scla.invpascal(n).astype(int) for n in n_values]
	tests['Pascal'] = PinvStabilityTester(matrices, pinv_ref)
	tests['Pascal'].compute_all()
	if LATEX_DISPLAY:
		tests['Pascal'].print_all(parameters=OrderedDict({'$n$': [r'${:d}$'.format(n) for n in n_values]}), latex=True, label='Pascal', caption='Pascal matrices $P(n)$')
	else:
		tests['Pascal'].print_all(parameters=OrderedDict({'n': n_values}))
	
	# N(0,1) random matrices
	if not LATEX_DISPLAY: print('Treating N(0,1) random matrices:')
	n_values = (4, 6, 8, 10)
	matrices = [reset_randn(3*n, n).astype(np.float) for n in n_values]
	pinv_ref = [scla.lstsq(matrix, np.eye(matrix.shape[0]), cond=rcond, lapack_driver='gelsy')[0] for matrix in matrices]
	tests['random'] = PinvStabilityTester(matrices, pinv_ref)
	tests['random'].compute_all()
	if LATEX_DISPLAY:
		tests['random'].print_all(parameters=OrderedDict({'$n$': [r'${:d}$'.format(n) for n in n_values]}), latex=True, label='random', caption=r'random matrices $R_\mathcal{N} \in \mathbb{R}^{3n\times n}$ with elements distributed from $\mathcal{N}(0,1)$')
	else:
		tests['random'].print_all(parameters=OrderedDict({'n': n_values}))
	
	# Random ill-conditioned matrices
	if not LATEX_DISPLAY: print('Treating random ill-conditioned matrices:')
	n_values = (6, 8, 10, 12)
	matrices = [np.linalg.matrix_power(reset_randn(n, n), 4).astype(np.float) for n in n_values]
	pinv_ref = [np.linalg.matrix_power(scla.lstsq(reset_randn(n, n), np.eye(n), cond=rcond, lapack_driver='gelsy')[0], 4) for n in n_values]
	tests['ill-cond'] = PinvStabilityTester(matrices, pinv_ref)
	tests['ill-cond'].compute_all()
	if LATEX_DISPLAY:
		tests['ill-cond'].print_all(parameters=OrderedDict({'$n$': [r'${:d}$'.format(n) for n in n_values]}), latex=True, label='ill-cond', caption=r'random ill-conditioned matrices $R_\mathcal{N}^4 \in \mathbb{R}^{n\times n}$')
	else:
		tests['ill-cond'].print_all(parameters=OrderedDict({'n': n_values}))
	
	# Matrix USV
	if not LATEX_DISPLAY: print('Treating USV matrices:')
	n_values = range(10, 21, 5) # (10, 25, 40, 55, 70)
	matrices = []
	pinv_ref = []
	norm_mat = []
	norm_pinv_ref = []
	cond_nb_mat = []
	d = np.sqrt(2)
	for n in n_values:
		U = reset_rando(5*n, n)
		V = reset_rando(n)
		S = [d**i for i in range(n)]
		matrices.append(np.dot(U*S, V.T))
		pinv_ref.append(np.dot(V/S, U.T))
		norm_mat.append(d**(n-1))
		cond_nb_mat.append(d**(n-1))
		norm_pinv_ref.append(1)
	tests['USV'] = PinvStabilityTester(matrices, pinv_ref, norm_mat=norm_mat, norm_pinv_ref=norm_pinv_ref, cond_nb_mat=cond_nb_mat, rcond=1e-8)
	tests['USV'].compute_all()
	if LATEX_DISPLAY:
		tests['USV'].print_all(parameters=OrderedDict({'$n$': [r'${:d}$'.format(n) for n in n_values]}), latex=True, label='USV', caption=r'random matrices $USV^\top \in \mathbb{R}^{5n\times n}$, where $U$ and $V$ are random column orthogonal matrices and $S = diag(1, 2^\frac{1}{2}, \ldots, 2^\frac{n-1}{2})$')
	else:
		tests['USV'].print_all(parameters=OrderedDict({'n': n_values}))
	
	# Kahan matrix
	if not LATEX_DISPLAY: print('Treating Kahan matrices:')
	c_values = np.arange(0.1, 0.45, 0.05)
	matrices = []
	pinv_ref = []
	norm_mat = []
	norm_pinv_ref = []
	cond_nb_mat = []
	n = 100
	I = np.arange(n)[:,None]; J = np.arange(n)
	for c in c_values:
		s = np.sqrt(1-c**2)
		S = np.array([s**i for i in range(n)])
		matrices.append((np.eye(n) - c*np.triu(np.ones((n,n)), 1))*S[:,None])
		pinv_ref.append((np.triu(c*(1+c)**np.clip(J-I-1,0,None), 1) + np.eye(n))/S)
	tests['Kahan'] = PinvStabilityTester(matrices, pinv_ref, rcond=1e-19)
	tests['Kahan'].compute_all()
	if LATEX_DISPLAY:
		tests['Kahan'].print_all(parameters=OrderedDict({'c': [r'${:.2f}$'.format(c) for c in c_values]}), latex=True, label='Kahan', caption=r'Kahan matrices $K(c, s) \in \mathbb{R}^{100\times 100}$, with $c^2 + s^2 = 1$')
	else:
		tests['Kahan'].print_all(parameters=OrderedDict({'c': ['{:.2f}\0'.format(c) for c in c_values]}))
