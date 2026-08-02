"""Molecule-specific adsorption models: 1,4-CHD and pentacene on Si(001)."""
from itertools import product

import numpy as np

from tenet.models import inf, register


@register("CHD_simple")
def chd_simple(calc, temp, m_par):
	"""1,4-cyclohexadiene on Si(001)-(2x1), simple variant.

	m_par: 0, 1 - chemical potentials of the two adsorption states the code
	calls t_sigma and d_sigma; 2..7 - the pair interaction energies.

	Drivers: 14CHD_Si_simple_tm.py, 14CHD_Si_simple_trg.py.
	"""
	neigbours = calc.coord
	mu_t_sigma = m_par[0] / neigbours
	mu_d_sigma = m_par[1] / neigbours
	e_d_d_in = -m_par[2]
	e_d_t_in = -m_par[3]
	e_t_t_in = -m_par[4]
	e_d_d_out = -m_par[5]
	e_d_t_out = -m_par[6]
	e_t_t_out = -m_par[7]
	matrixes = [np.array([[0, mu_d_sigma, mu_t_sigma / 2.0, inf], \
					[mu_d_sigma, mu_d_sigma * 2.0 + e_d_d_in, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_in, inf], \
					[inf, inf, inf, mu_t_sigma], \
					[mu_t_sigma / 2.0, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_in, mu_t_sigma + e_t_t_in, inf]]), \
				np.array([[0, mu_d_sigma, mu_t_sigma / 2.0, mu_t_sigma / 2.0], \
					[mu_d_sigma, mu_d_sigma * 2.0 + e_d_d_out, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_out, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_out], \
					[mu_t_sigma / 2.0, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_out, mu_t_sigma + e_t_t_out / 2.0, mu_t_sigma], \
					[mu_t_sigma / 2.0, mu_d_sigma + mu_t_sigma / 2.0 + e_d_t_out, mu_t_sigma, mu_t_sigma + e_t_t_out / 2.0]])]
	return matrixes


@register("CHD_complex")
def chd_complex(calc, temp, m_par):
	"""1,4-cyclohexadiene on Si(001)-(2x1), extended state space.

	m_par: 0, 1 - the t_sigma and d_sigma chemical potentials; 2..12 - the
	pair interaction energies of the richer configuration set.

	Drivers: 14CHD_Si_complex_tm.py, 14CHD_Si_complex_trg.py.
	"""
	neigbours = calc.coord
	mu_t_sigma = m_par[0] / neigbours
	mu_d_sigma = m_par[1] / neigbours
	e_d_d_hor_same = -m_par[2]
	e_d_d_hor_dif_in = -m_par[3]
	e_d_d_hor_dif_out = -m_par[4]
	e_d_t_hor_in = -m_par[5]
	e_d_t_hor_out = -m_par[6]
	e_t_t_hor = -m_par[7]

	e_d_d_vert_same = -m_par[8]
	e_d_d_vert_dif = -m_par[9]
	e_d_t_vert_in = -m_par[10]
	e_d_t_vert_out = -m_par[11]
	e_t_t_vert = -m_par[12]
	matrixes = [np.array([[0, mu_t_sigma / 2.0, inf, mu_d_sigma, mu_d_sigma], \
					[inf, inf, mu_t_sigma, inf, inf], \
					[mu_t_sigma / 2.0, mu_t_sigma + e_t_t_hor, inf, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_hor_in, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_hor_out], \
					[mu_d_sigma, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_hor_out, inf, mu_d_sigma * 2.0 + e_d_d_hor_same, mu_d_sigma * 2.0 + e_d_d_hor_dif_out], \
					[mu_d_sigma, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_hor_in, inf, mu_d_sigma * 2.0 + e_d_d_hor_dif_in, mu_d_sigma * 2.0 + e_d_d_hor_same]]), \
				np.array([[0, mu_t_sigma / 2.0, mu_t_sigma / 2.0, mu_d_sigma, mu_d_sigma], \
					[mu_t_sigma / 2.0, mu_t_sigma + e_t_t_vert / 2.0, mu_t_sigma, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_out, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_in], \
					[mu_t_sigma / 2.0, mu_t_sigma, mu_t_sigma + e_t_t_vert / 2.0, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_in, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_out], \
					[mu_d_sigma, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_out, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_in, mu_d_sigma * 2.0 + e_d_d_vert_same, mu_d_sigma * 2.0 + e_d_d_vert_dif], \
					[mu_d_sigma, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_in, mu_t_sigma / 2.0 + mu_d_sigma + e_d_t_vert_out, mu_d_sigma * 2.0 + e_d_d_vert_dif, mu_d_sigma * 2.0 + e_d_d_vert_same]])]
	return matrixes


@register("Pentacene_model_1_simple")
def pentacene_model_1_simple(calc, temp, m_par):
	"""Pentacene on Si(001)-2x1, model 1, simple variant: 5 states, 3 merged sites.

	m_par: 0 - mu, 1 - the close-contact energy, 2, 3 - the two longer-range
	pair energies.

	Driver: Pentacene_model_1_simple_tm.py.
	"""
	neigbours = calc.coord
	mu = m_par[0] / neigbours
	e_close = -m_par[1]
	e_one = -m_par[2]
	e_two = -m_par[3]
	states = 5
	nodes = 3
	calc.nodes = nodes
	exist = [[1, 1, 0, 0, 0], \
					[0, 0, 1, 0, 0], \
					[0, 0, 0, 1, 0], \
					[0, 0, 0, 0, 1], \
					[1, 1, 0, 0, 0]]
	energies = [[0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0], \
					[0, e_close, 0, 0, 0]]
	#combination with e_one energy
	energy_one = [4, 0, 1]
	#combination with e_two energy
	energy_two = [4, 0, 0, 1]
	chem = [0, mu / 4.0, mu / 4.0, mu / 4.0, mu / 4.0]
	all_combinations = product(range(states), repeat = nodes)
	combinations = []
	combinations_mu = []
	combinations_en = []
	for cur in all_combinations:
		cur_mu = chem[cur[0]]
		cur_en = 0
		comb_no = False
		for i in range(nodes - 1):
			cur_mu += chem[cur[i + 1]]
			cur_en += energies[cur[i]][cur[i + 1]] / 2.0
			if exist[cur[i]][cur[i + 1]] == 0:
				comb_no = True
		if comb_no:
			continue
		if len(cur) > 2:
			for i in range(len(cur) - 2):
				if list(cur[i:i + 3]) == list(energy_one):
					cur_en += e_one / 2.0
		if len(cur) > 3:
			for i in range(len(cur) - 3):
				if list(cur[i:i + 4]) == list(energy_two):
					cur_en += e_two / 2.0
		combinations.append(cur)
		combinations_mu.append(cur_mu)
		combinations_en.append(cur_en)
	mat1 = []
	mat2 = []
	for l_num, left in enumerate(combinations):
		line = []
		line2 = []
		for r_num, right in enumerate(combinations):
			cur_mu = combinations_mu[l_num] + combinations_mu[r_num]
			if exist[left[-1]][right[0]] == 0:
				line.append(inf)
				line2.append(cur_mu)
				continue
			cur_en = combinations_en[l_num] + combinations_en[r_num] + energies[left[-1]][right[0]]
			cur = left + right
			for i in range(2):
				if i + 3 > len(cur):
					break
				comp_list_one = list(cur[nodes - 2 + i:nodes - 2 + 3 + i])
				if comp_list_one == energy_one:
					cur_en += e_one
			for i in range(3):
				if i + 4 > len(cur):
					break
				comp_list_two = list(cur[nodes - 3 + i:nodes - 3 + 4 + i])
				if comp_list_two == energy_two:
					cur_en += e_two
			line.append(cur_mu + cur_en)
			line2.append(cur_mu)
		mat1.append(line)
		mat2.append(line2)
	matrixes = [np.array(mat1), np.array(mat2)]
	return matrixes


@register("Pentacene_model_1_complex")
def pentacene_model_1_complex(calc, temp, m_par):
	"""Pentacene on Si(001)-2x1, model 1, extended: 9 states, 3 merged sites.

	m_par: 0, 1 - the perpendicular and parallel chemical potentials;
	2..13 - the pair energies.

	Driver: Pentacene_model_1_complex_tm.py.
	"""
	neigbours = calc.coord
	mu_pentacene_per = m_par[0] / neigbours / 4.0
	mu_pentacene_par = m_par[1] / neigbours / 4.0
	e_1 = -m_par[2]
	e_2 = -m_par[3]
	e_3 = -m_par[4]
	e_4 = -m_par[5]
	e_5 = -m_par[6]
	e_6 = -m_par[7]
	e_7 = -m_par[8]
	e_8 = -m_par[9]
	e_9 = -m_par[10]
	e_10 = -m_par[11]
	e_11 = -m_par[12]
	e_12 = -m_par[13]
	states = 9
	nodes = 3
	calc.nodes = nodes
	exist = [[1, 1, 0, 0, 0, 1, 0, 1, 0], \
					[0, 0, 1, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 1, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 1, 0, 0, 0, 0], \
					[1, 1, 0, 0, 0, 1, 0, 1, 0], \
					[0, 0, 0, 0, 0, 0, 1, 0, 0], \
					[1, 1, 0, 0, 0, 1, 0, 1, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 1], \
					[1, 1, 0, 0, 0, 1, 0, 1, 0]]
	energies = [[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_1, 0, 0, 0, e_10, 0, e_10, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_10, 0, 0, 0, e_4, 0, e_7, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_10, 0, 0, 0, e_7, 0, e_4, 0]]
	energy_one = {str([4, 0, 1]): e_2, \
			str([4, 0, 5]): e_11, \
			str([4, 0, 7]): e_11, \
			str([6, 0, 1]): e_11, \
			str([8, 0, 1]): e_11, \
			str([6, 0, 5]): e_5, \
			str([6, 0, 7]): e_8, \
			str([8, 0, 5]): e_8, \
			str([8, 0, 7]): e_5}
	#combination with e_two energy
	energy_two = {str([4, 0, 0, 1]): e_3, \
			str([4, 0, 0, 5]): e_12, \
			str([4, 0, 0, 7]): e_12, \
			str([6, 0, 0, 1]): e_12, \
			str([8, 0, 0, 1]): e_12, \
			str([6, 0, 0, 5]): e_6, \
			str([6, 0, 0, 7]): e_9, \
			str([8, 0, 0, 5]): e_9, \
			str([8, 0, 0, 7]): e_6}
	chem = [0, mu_pentacene_par, mu_pentacene_par, mu_pentacene_par, mu_pentacene_par, \
			mu_pentacene_per, mu_pentacene_per, mu_pentacene_per, mu_pentacene_per]
	all_combinations = product(range(states), repeat = nodes)
	combinations = []
	combinations_mu = []
	combinations_en = []
	for cur in all_combinations:
		cur_mu = chem[cur[0]]
		cur_en = 0
		comb_no = False
		for i in range(nodes - 1):
			cur_mu += chem[cur[i + 1]]
			cur_en += energies[cur[i]][cur[i + 1]] / 2.0
			if exist[cur[i]][cur[i + 1]] == 0:
				comb_no = True
		if comb_no:
			continue
		if len(cur) > 2:
			for i in range(len(cur) - 2):
				cur_en += energy_one.get(str(list(cur[i:i + 3])), 0) / 2.0
		if len(cur) > 3:
			for i in range(len(cur) - 3):
				cur_en += energy_two.get(str(list(cur[i:i + 4])), 0) / 2.0
		combinations.append(cur)
		combinations_mu.append(cur_mu)
		combinations_en.append(cur_en)
	mat1 = []
	mat2 = []
	for l_num, left in enumerate(combinations):
		line = []
		line2 = []
		for r_num, right in enumerate(combinations):
			cur_mu = combinations_mu[l_num] + combinations_mu[r_num]
			if exist[left[-1]][right[0]] == 0:
				line.append(inf)
				line2.append(cur_mu)
				continue
			cur_en = combinations_en[l_num] + combinations_en[r_num] + energies[left[-1]][right[0]]
			cur = left + right
			for i in range(2):
				if i + 3 > len(cur):
					break
				comp_list_one = list(cur[nodes - 2 + i:nodes - 2 + 3 + i])
				if comp_list_one == energy_one:
					cur_en += e_one
			for i in range(3):
				if i + 4 > len(cur):
					break
				comp_list_two = list(cur[nodes - 3 + i:nodes - 3 + 4 + i])
				if comp_list_two == energy_two:
					cur_en += e_two
			line.append(cur_mu + cur_en)
			line2.append(cur_mu)
		mat1.append(line)
		mat2.append(line2)
	matrixes = [np.array(mat1), np.array(mat2)]
	return matrixes


@register("Pentacene_model_3")
def pentacene_model_3(calc, temp, m_par):
	"""Pentacene on Si(001)-2x1, model 3.

	m_par: 0, 1 - the perpendicular and parallel chemical potentials;
	2..14 - the pair energies.

	Drivers: Pentacene_model_3_trg.py, Pentacene_model_3_trg_v.2.py.
	"""
	neigbours = calc.coord
	mu_pentacene_per = m_par[0] / neigbours / 4.0
	mu_pentacene_par = m_par[1] / neigbours / 4.0
	e_v1 = -m_par[2] / 4.0
	e_v2 = -m_par[3] / 3.0
	e_v3 = -m_par[4] / 2.0
	e_v4 = -m_par[5] / 2.0
	e_v5 = -m_par[6] / 2.0
	e_v6 = -m_par[7]
	e_v7 = -m_par[8] / 2.0
	e_h1 = -m_par[9] / 2.0
	e_h2 = -m_par[10]
	e_h3 = -m_par[11]
	e_h4 = -m_par[12]
	e_v8 = -m_par[13]
	e_v9 = -m_par[14]
	chem_pot = np.array([0, mu_pentacene_par, mu_pentacene_par, mu_pentacene_par, mu_pentacene_par, mu_pentacene_per, mu_pentacene_per, mu_pentacene_per, mu_pentacene_per])
	matrixes = [np.array([[0, 0, inf, inf, inf, 0, inf, 0, inf], \
					[inf, inf, 0, inf, inf, inf, inf, inf, inf], \
					[inf, inf, inf, 0, inf, inf, inf, inf, inf], \
					[inf, inf, inf, inf, 0, inf, inf, inf, inf], \
					[0, e_h4, inf, inf, inf, e_h3, inf, e_h3, inf], \
					[inf, inf, inf, inf, inf, inf, 0, inf, inf], \
					[0, e_h3, inf, inf, inf, e_h1, inf, e_h2, inf], \
					[inf, inf, inf, inf, inf, inf, inf, inf, 0], \
					[0, e_h3, inf, inf, inf, e_h2, inf, e_h1, inf]]), \
					np.array([[0, 0, 0, 0, 0, 0, 0, inf, inf], \
					[0, e_v1, e_v2, e_v3, e_v8, e_v5, e_v6, inf, inf], \
					[0, e_v2, e_v1, e_v2, e_v3, e_v4, e_v5, inf, inf], \
					[0, e_v3, e_v2, e_v1, e_v2, e_v5, e_v4, inf, inf], \
					[0, e_v8, e_v3, e_v2, e_v1, e_v6, e_v5, inf, inf], \
					[inf, inf, inf, inf, inf, inf, inf, 0, inf], \
					[inf, inf, inf, inf, inf, inf, inf, inf, 0], \
					[0, e_v5, e_v4, e_v5, e_v6, e_v7, e_v9, inf, inf], \
					[0, e_v6, e_v5, e_v4, e_v5, e_v9, e_v7, inf, inf]])]
	for i in range(len(matrixes[0][0])):
		for j in range(len(matrixes[0][0])):
			if (matrixes[0][i][j] - 0.1) > inf:
				matrixes[0][i][j] += chem_pot[i] + chem_pot[j]
			if (matrixes[1][i][j] - 0.1) > inf:
				matrixes[1][i][j] += chem_pot[i] + chem_pot[j]

	return matrixes
