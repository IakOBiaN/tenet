"""Pentacene on Si(001)-2x1, model 2 - the largest of the adsorption models."""
from itertools import product

import numpy as np

from tenet.models import inf, register


@register("Pentacene_model_2")
def pentacene_model_2(calc, temp, m_par):
	"""Pentacene model 2: 15 states over 3 merged sites, 48 parameters.

	m_par: 0, 1, 2 - the chemical potentials of the three adsorption
	orientations; 3..47 - six blocks of pair energies (e_44, e_22, e_33, e_24,
	e_34, e_23).

	Of the 15**3 state triples only those allowed by the ``exist`` adjacency
	table survive, which leaves 100 - the matrices are 100x100, not 3375x3375.

	Driver: Pentacene_model_2_tm.py.
	"""
	neigbours = calc.coord
	mu_2 = m_par[0] / neigbours / 2.0
	mu_3 = m_par[1] / neigbours / 3.0
	mu_4 = m_par[2] / neigbours / 4.0

	e_44_1 = -m_par[3]
	e_44_2 = -m_par[4]
	e_44_3 = -m_par[5]

	e_22_1 = -m_par[6]
	e_22_2 = -m_par[7]
	e_22_3 = -m_par[8]
	e_22_4 = -m_par[9]
	e_22_5 = -m_par[10]
	e_22_6 = -m_par[11]
	e_22_7 = -m_par[12]
	e_22_8 = -m_par[13]
	e_22_9 = -m_par[14]

	e_33_1 = -m_par[15]
	e_33_2 = -m_par[16]
	e_33_3 = -m_par[17]
	e_33_4 = -m_par[18]
	e_33_5 = -m_par[19]
	e_33_6 = -m_par[20]
	e_33_7 = -m_par[21]
	e_33_8 = -m_par[22]
	e_33_9 = -m_par[23]

	e_24_1 = -m_par[24]
	e_24_2 = -m_par[25]
	e_24_3 = -m_par[26]
	e_24_4 = -m_par[27]
	e_24_5 = -m_par[28]
	e_24_6 = -m_par[29]

	e_34_1 = -m_par[30]
	e_34_2 = -m_par[31]
	e_34_3 = -m_par[32]
	e_34_4 = -m_par[33]
	e_34_5 = -m_par[34]
	e_34_6 = -m_par[35]

	e_23_1 = -m_par[36]
	e_23_2 = -m_par[37]
	e_23_3 = -m_par[38]
	e_23_4 = -m_par[39]
	e_23_5 = -m_par[40]
	e_23_6 = -m_par[41]
	e_23_7 = -m_par[42]
	e_23_8 = -m_par[43]
	e_23_9 = -m_par[44]
	e_23_10 = -m_par[45]
	e_23_11 = -m_par[46]
	e_23_12 = -m_par[47]

	states = 15
	nodes = 3
	calc.nodes = nodes
	exist = [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], \
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], \
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], \
					[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], \
					[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
					[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]]
	energies = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_22_1, 0, e_22_3, 0, e_23_3, 0, 0, e_23_1, 0, 0, e_24_2, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_22_2, 0, e_22_1, 0, e_23_4, 0, 0, e_23_2, 0, 0, e_24_1, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_23_2, 0, e_23_1, 0, e_33_1, 0, 0, e_33_2, 0, 0, e_34_1, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_23_4, 0, e_23_3, 0, e_33_3, 0, 0, e_33_1, 0, 0, e_34_2, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
					[0, e_24_1, 0, e_24_2, 0, e_34_2, 0, 0, e_34_1, 0, 0, e_44_1, 0, 0, 0]]
	#combination with e_one energy
	energy_one = {str([14, 0, 11]): e_44_2, \
			str([2, 0, 1]): e_22_4, \
			str([4, 0, 3]): e_22_4, \
			str([4, 0, 1]): e_22_6, \
			str([2, 0, 3]): e_22_8, \
			str([7, 0, 5]): e_33_4, \
			str([10, 0, 8]): e_33_4, \
			str([7, 0, 8]): e_33_6, \
			str([10, 0, 5]): e_33_8, \
			str([14, 0, 1]): e_24_4, \
			str([4, 0, 11]): e_24_4, \
			str([14, 0, 3]): e_24_6, \
			str([2, 0, 11]): e_24_6, \
			str([14, 0, 8]): e_34_3, \
			str([7, 0, 11]): e_34_3, \
			str([14, 0, 5]): e_34_5, \
			str([10, 0, 11]): e_34_5, \
			str([7, 0, 3]): e_23_5, \
			str([2, 0, 8]): e_23_5, \
			str([7, 0, 1]): e_23_7, \
			str([4, 0, 8]): e_23_7, \
			str([2, 0, 5]): e_23_9, \
			str([10, 0, 3]): e_23_9, \
			str([4, 0, 5]): e_23_11, \
			str([10, 0, 1]): e_23_11}
	#combination with e_two energy
	energy_two = {str([14, 0, 0, 11]): e_44_3, \
			str([2, 0, 0, 1]): e_22_5, \
			str([4, 0, 0, 3]): e_22_5, \
			str([4, 0, 0, 1]): e_22_7, \
			str([2, 0, 0, 3]): e_22_9, \
			str([7, 0, 0, 5]): e_33_5, \
			str([10, 0, 0, 8]): e_33_5, \
			str([7, 0, 0, 8]): e_33_7, \
			str([10, 0, 0, 5]): e_33_9, \
			str([14, 0, 0, 3]): e_24_3, \
			str([2, 0, 0, 11]): e_24_3, \
			str([14, 0, 0, 1]): e_24_5, \
			str([4, 0, 0, 11]): e_24_5, \
			str([14, 0, 0, 8]): e_34_4, \
			str([7, 0, 0, 11]): e_34_4, \
			str([14, 0, 0, 5]): e_34_6, \
			str([10, 0, 0, 11]): e_34_6, \
			str([7, 0, 0, 3]): e_23_6, \
			str([2, 0, 0, 8]): e_23_6, \
			str([7, 0, 0, 1]): e_23_8, \
			str([4, 0, 0, 8]): e_23_8, \
			str([2, 0, 0, 5]): e_23_10, \
			str([10, 0, 0, 3]): e_23_10, \
			str([4, 0, 0, 5]): e_23_12, \
			str([10, 0, 0, 1]): e_23_12}
	chem = [0, mu_2, mu_2, mu_2, mu_2, \
			mu_3, mu_3, mu_3, mu_3, mu_3, mu_3, \
			mu_4, mu_4, mu_4, mu_4]
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
				cur_en += energy_one.get(str(comp_list_one), 0)
			for i in range(3):
				if i + 4 > len(cur):
					break
				comp_list_two = list(cur[nodes - 3 + i:nodes - 3 + 4 + i])
				cur_en += energy_two.get(str(comp_list_two), 0)
			line.append(cur_mu + cur_en)
			line2.append(cur_mu)
		mat1.append(line)
		mat2.append(line2)
	matrixes = [np.array(mat1), np.array(mat2)]
	return matrixes
