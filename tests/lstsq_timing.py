#!/usr/bin/env python3

import timeit
from inspect import cleandoc as trim
import numpy as np
import scipy.linalg as scla
from context import sample as rank_greville
from unit_testing import generate_normal_dist_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

### Tools

class TimerCustom(timeit.Timer):
	def custom_autorange(self, threshold=10, ini=1, factor=2, max_N=1000000):
		N = ini
		while True:
			timing = self.timeit(number=N)
			if timing > threshold or N >= max_N:
				break
			N *= factor
		return(N, timing)

def autorangemin(stmt, setup, repeat=10):
	timer = TimerCustom(stmt, setup)
	N, min_time = timer.custom_autorange(max_N=1)
	if repeat > 1:
		min_time = min(min_time, min(timer.repeat(repeat=repeat-1, number=N)))
	return(min_time/N, N)

def dict_append_update(old_dict, new_dict, value=None):
	for key, item in new_dict.items():
		if value is not None:
			item = value
		if key in old_dict:
			old_dict[key].append(item)
		else:
			old_dict[key] = [item]

### Tests

def run_all_lstsq(n, m, r, disabled='', threshold=10, repeats=10):
	timing = dict()
	repeat = dict()
	for algo in 'gelsy', 'gelsd', 'gelss':
		if algo not in disabled:
			setup = '''from __main__ import scla, generate_normal_dist_matrix
				   A = generate_normal_dist_matrix({n}, {m}, {r})
				   b = generate_normal_dist_matrix({n}, 1, {n}).flatten()'''.format(n=n, m=m, r=r)
			stmt = '''scla.lstsq(A, b, lapack_driver='{}')'''.format(algo)
			timing[algo], repeat[algo] = autorangemin(stmt, trim(setup), repeat=repeats)
			if timing[algo] > threshold:
				disabled.add(algo)
	
	if 'rank-Greville' not in disabled:
		setup = '''from __main__ import pseudoinverse, generate_normal_dist_matrix
			   A = generate_normal_dist_matrix({n}, {m}, {r})
			   b = generate_normal_dist_matrix({n}, 1, {n}).flatten()
			   model = rank_greville.RecursiveModel()'''.format(n=n, m=m, r=r)
		stmt = '''model.add_observations(A, b)'''
		timing['rank-Greville'], repeat['rank-Greville'] = autorangemin(stmt, trim(setup), repeat=repeats)
		if timing['rank-Greville'] > threshold:
			disabled.add('rank-Greville')
		
#		A = generate_normal_dist_matrix(n, m, r)
#		b = generate_normal_dist_matrix(n, 1, n).flatten()
#		model = rank_greville.RecursiveModel()
#		model.add_observations(A, b)
#		lapack_parameters = scla.lstsq(A, b, lapack_driver='gelsy', cond=1e-10)[0]
#		print('maximum error:', np.max(np.abs(lapack_parameters - model.parameters)))
	
	return(timing, repeat)

def plot_asympt(timings, ranges, variable='n', filename=None, nb_lasts=dict(), slicing=slice(None)):
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(111)
	for algo, timing in timings.items():
		timing = timing[slicing]
		n_range = ranges[algo][slicing]
		if algo in nb_lasts:
			nb_last = nb_lasts[algo]
		else:
			nb_last = round(len(n_range)/3)
		coefs = np.polyfit(np.log(n_range[-nb_last:]), np.log(timing[-nb_last:]), 1)
		deg, base = coefs
		ax.scatter(n_range, timing)
		ax.plot(n_range, np.exp(np.polyval(coefs, np.log(n_range))), label=r'{}: ${}^{{{:.2f}}}$'.format(algo, variable, deg))
	ax.set_xlabel(variable, fontsize=14)
	ax.set_ylabel('elapsed time (s)', fontsize=14)
	ax.legend(fontsize=12)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.tick_params(labelsize=12)
	fig.suptitle('Timing for solving the linear least-squares problem', fontsize=16)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)

def plot_exponent(timings, ranges, variable='n', filename=None, nb_lasts=dict()):
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(111)
	for algo, timing in timings.items():
		n_range = ranges[algo]
		if algo in nb_lasts:
			nb_last = nb_lasts[algo]
		else:
			nb_last = int(round(len(n_range)/3))
		exponent_instant = np.gradient(np.log(timing))/np.gradient(np.log(n_range))
		deg = np.mean(exponent_instant[-nb_last:])
		ax.scatter(n_range, exponent_instant)
		ax.plot(n_range, [deg]*len(n_range), label=r'{}: ${}^{{{:.2f}}}$'.format(algo, variable, deg))
	ax.set_xlabel(variable, fontsize=14)
	ax.set_ylabel('dlog(t)/dlog(n)', fontsize=14)
	ax.legend(fontsize=12)
	ax.tick_params(labelsize=12)
	fig.suptitle('Time complexity exponent for solving the linear least-squares problem', fontsize=16)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)

def plot_domains(points, timings, grid_nb=100, filename='Domains_ratios.png'):
	points = np.array(points)
	
	sorted_timings = np.sort(np.array([values for values in timings.values() if len(values) > 0]), axis=0)
	diff = sorted_timings[1]/sorted_timings[0]
	grid_x, grid_y = np.meshgrid(np.linspace(min(points[:,0]), max(points[:,0]), grid_nb), np.linspace(min(points[:,1]), max(points[:,1]), grid_nb))
	interpolation = griddata(points, diff, (grid_x, grid_y), method='linear')
	
	best_only = dict()
	for algo, timing in timings.items():
		try:
			best_only[algo] = points[timing == sorted_timings[0]].T
		except Exception:
			best_only[algo] = [[],[]]
	
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(111)
	
	for algo, coords in best_only.items():
		ax.scatter(*coords, label=algo, s=200, zorder=3)
	
	levels = [1.25, 1.4, 1.8, 2, 3, 4]
	CS = ax.contour(grid_x, grid_y, interpolation, levels=levels, alpha=0.5) #, label='iterpolated margin')
	ax.contourf(grid_x, grid_y, interpolation, levels=levels, extend='max', alpha=0.05) #, label='iterpolated margin')
	ax.clabel(CS, fmt='x%1.2f') #, fontsize=11)
	
	ax.set_xlabel('$n/m$ ratio', fontsize=14)
	ax.set_ylabel('$r/n$ ratio', fontsize=14)
	ax.legend(fontsize=12)
	ax.tick_params(labelsize=12)
	fig.suptitle('Most efficient algorithm and relative margin', fontsize=16) # Most efficient algorithm depending on asymptotic parameters ratios
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)

if __name__ == '__main__':
	do_squares = False
	do_rect = False
	do_rank = False
	do_domains = False
	nb_retry = 10
	
	# Square matrices
	print('Treating square matrices')
	if do_squares:
		timings_square = dict()
		repeats_square = dict()
		ranges_square = dict()
		disabled = set()
		n_range = np.linspace(1e2, 1e4, 100).astype(int)
		for n in n_range:
			print('n = {n}'.format(n=n))
			new_timing, new_repeat = run_all_lstsq(n, n, n, disabled, repeats=nb_retry)
			print('timing (best of {}): '.format(nb_retry), new_timing)
#			print('repeats: ', new_repeat)
			dict_append_update(timings_square, new_timing)
			dict_append_update(repeats_square, new_repeat)
			dict_append_update(ranges_square, new_timing, value=n)
		with open('square_timings.pickle', 'wb') as f:
			pickle.dump((timings_square, repeats_square, ranges_square), f)
	else:
		with open('square_timings.pickle', 'rb') as f:
			(timings_square, repeats_square, ranges_square) = pickle.load(f)
	plot_asympt(timings_square, ranges_square, variable='n', filename='Cost_full_rank_square.png', nb_lasts={'rank-Greville': 2, 'gelss': 3})
	
	# Rectangular matrices (fixed nb of rows)
	print('Treating rectangular matrices with fixed rows nb')
	if do_rect:
		timings_rect = dict()
		repeats_rect = dict()
		ranges_rect = dict()
		disabled = set()
		m_range = np.linspace(1e3, 1e5, 100).astype(int)
		n = 100
		for m in m_range:
			r = min(m, n)
			print('m = {m}'.format(m=m))
			new_timing, new_repeat = run_all_lstsq(n, m, r, disabled, repeats=nb_retry)
			print('timing (best of {}): '.format(nb_retry), new_timing)
#			print('repeats: ', new_repeat)
			dict_append_update(timings_rect, new_timing)
			dict_append_update(repeats_rect, new_repeat)
			dict_append_update(ranges_rect, new_timing, value=m)
		with open('rect_timings.pickle', 'wb') as f:
			pickle.dump((timings_rect, repeats_rect, ranges_rect), f)
	else:
		with open('rect_timings.pickle', 'rb') as f:
			(timings_rect, repeats_rect, ranges_rect) = pickle.load(f)
	plot_asympt(timings_rect, ranges_rect, variable='m', filename='Cost_fixed_rows.png', slicing=slice(1, None, None))
	
	# Rank-deficient square matrices (fixed rank)
	print('Treating rectangular matrices with fixed rank')
	if do_rank:
		timings_rank = dict()
		repeats_rank = dict()
		ranges_rank = dict()
		disabled = set()
		n_range = np.linspace(1e2, 1e4, 100).astype(int)
		r = 100
		for n in n_range:
			print('n = {n}'.format(n=n))
			new_timing, new_repeat = run_all_lstsq(n, n, r, disabled, repeats=nb_retry)
			print('timing (best of {}): '.format(nb_retry), new_timing)
#			print('repeats: ', new_repeat)
			dict_append_update(timings_rank, new_timing)
			dict_append_update(repeats_rank, new_repeat)
			dict_append_update(ranges_rank, new_timing, value=n)
		with open('rank_timings.pickle', 'wb') as f:
			pickle.dump((timings_rank, repeats_rank, ranges_rank), f)
	else:
		with open('rank_timings.pickle', 'rb') as f:
			(timings_rank, repeats_rank, ranges_rank) = pickle.load(f)
	plot_asympt(timings_rank, ranges_rank, variable='n', filename='Cost_fixed_rank.png')
	
	# Ratios domains
	print('Treating ratio domains')
	if do_domains:
		timings_ratio = dict()
		repeats_ratio = dict()
		disabled = set()
		ratios = np.dstack(np.meshgrid(np.linspace(0.1, 1, 10)[::-1], np.linspace(0.03, 0.3, 10)[::-1])).reshape(-1, 2)
		m = 4000
		for ratio in ratios:
			n = int(round(ratio[0] * m))
			r = int(round(ratio[1] * n))
			print('m = {m}, n = {n}, r = {r}'.format(m=m, n=n, r=r))
			new_timing, new_repeat = run_all_lstsq(n, m, r, disabled=disabled, repeats=nb_retry)
			disabled = set()
			print('timing (best of {}): '.format(nb_retry), new_timing)
#			print('repeats: ', new_repeat)
			dict_append_update(timings_ratio, new_timing)
			dict_append_update(repeats_ratio, new_repeat)
		with open('ratio_timings.pickle', 'wb') as f:
			pickle.dump((timings_ratio, repeats_ratio, ratios), f)
	else:
		with open('ratio_timings.pickle', 'rb') as f:
			(timings_ratio, repeats_ratio, ratios) = pickle.load(f)
	plot_domains(ratios, timings_ratio)
