"""Orientation-dependent long-range interactions on a triangular lattice."""
from itertools import product

import numpy as np

from tenet.models import inf, register


@register("2D_long-range_V")
def two_dim_long_range_v(calc, temp, m_par):
	"""As 2D_long-range, but the interaction follows from the particle shape.

	m_par: 0 - mu, 1 - (sigma, eps) of the radial term.

	Each particle carries three arms; a pair energy sums a radial part over
	the first three neighbour shells and an angular part over all arm-arm
	pairs, counting only arms that face one another.  Like 2D_long-range it
	merges triangular blocks and switches the lattice to the square algorithm.

	Driver: 2D_long_range_V.py.
	"""
	mu = m_par[0]
	chem = [0, mu, mu]
	parameters_of_the_model = m_par[1]

	#Vitaly's block for interactions calculation
	sigma = parameters_of_the_model[0]
	eps = parameters_of_the_model[1]

	distances = np.array([1, 3 ** 0.5, 2, 7 ** 0.5, 3, 2 * 3 ** 0.5, 13 ** 0.5, 4])
	angles = np.array([0, 0, 60.])
	w = -1.
	NN = 0
	i = 0
	j = 0
	en = 0
	interactions_list_1_right = []
	interactions_list_2_right = []
	interactions_list_3_right = []
	for NN in range(3):
		r = distances[NN]
		for i in range(3):
			line_1 = []
			line_2 = []
			line_3 = []
			for j in range(3):
				en = 0
				e_rad = 0
				e_ang = 0
				if i != 0 and j != 0:
					e_rad = eps * (5 * (sigma / r) ** 12 - 6 * (sigma / r) ** 10)
					for arm_i in range(3):
						for arm_j in range(3):
							alpha_i = angles[i] + (120 * arm_i)
							alpha_j = angles[j] + (120 * arm_j)
							en = 0
							alpha_i = alpha_i * np.pi / 180
							alpha_j = (180-alpha_j) * np.pi / 180
							if np.cos(alpha_i) < 0 or np.cos(alpha_j) < 0:
								en = 0
							else:
								en = (np.cos(alpha_i)**2) * (np.cos(alpha_j)**2)
							e_ang = e_ang + w * en
					en = e_rad + e_ang
				else:
					en = 0
				if NN == 0:
					line_1.append(en)
				if NN == 1:
					line_2.append(en)
				if NN == 2:
					line_3.append(en)
			if NN == 0:
				interactions_list_1_right.append(line_1)
			if NN == 1:
				interactions_list_2_right.append(line_2)
			if NN == 2:
				interactions_list_3_right.append(line_3)
	#print(np.array(interactions_list_1_right))

	interactions_list_1_right = np.array(interactions_list_1_right)
	interactions_list_1_diff = interactions_list_1_right.copy().T

	interactions_list_2_right = np.array(interactions_list_2_right)
	interactions_list_2_diff = interactions_list_2_right.copy().T

	interactions_list_3_right = np.array(interactions_list_3_right)
	interactions_list_3_diff = interactions_list_3_right.copy().T

	#print(interactions_list_1_right)
	#print(interactions_list_1_diff)
	#exit()

	number_of_interactions = 3
	triangular_number_order = 3
	triangular_number = round(1 / 2 * triangular_number_order * (triangular_number_order + 1)) #here we should calculate number of merged nodes
	past_triangular_number = round(1 / 2 * (triangular_number_order - 1) * ((triangular_number_order - 1) + 1))

	#calculate the total number of nodes
	nodes_for_chem_pot = ((triangular_number - past_triangular_number) * 2 + past_triangular_number) * 2
	#numer of edges in the new node + 1 (i don't know why)
	edges_with_bonds = 0
	for i in range(1, (triangular_number_order - 1) + 1):
			edges_with_bonds += 3 * i
	edges_with_bonds += 1

	calc.nodes = 1
	states = len(chem)

	new_nodes_combinations = product(range(states), repeat = triangular_number)
	combinations = []
	combinations_mu = []
	combinations_en = []
	#here we should create pattern for interactions
	pattern = np.zeros((11, 11), dtype = int)
	pattern_center = 5
	
	pattern[pattern_center][pattern_center] = 0
	pattern[pattern_center - 1][pattern_center] = 1
	pattern[pattern_center + 1][pattern_center] = 1
	pattern[pattern_center][pattern_center - 1] = 1
	pattern[pattern_center][pattern_center + 1] = 1
	pattern[pattern_center - 1][pattern_center - 1] = 1
	pattern[pattern_center + 1][pattern_center + 1] = 1

	pattern[pattern_center - 2][pattern_center - 1] = 2
	pattern[pattern_center - 1][pattern_center - 2] = 2
	pattern[pattern_center + 1][pattern_center - 1] = 2
	pattern[pattern_center + 2][pattern_center + 1] = 2
	pattern[pattern_center + 1][pattern_center + 2] = 2
	pattern[pattern_center - 1][pattern_center + 1] = 2

	pattern[pattern_center][pattern_center - 2] = 3
	pattern[pattern_center + 2][pattern_center] = 3
	pattern[pattern_center + 2][pattern_center + 2] = 3
	pattern[pattern_center][pattern_center + 2] = 3
	pattern[pattern_center - 2][pattern_center] = 3
	pattern[pattern_center - 2][pattern_center - 2] = 3

	#IMPORTATNT!!! For a triangular lattice we will use a square algorithm. I think this is a temporary solution.
	calc.lattice = "square"

	def add_en(i, j, ii, jj, comb_np, interactions_list):
		n1 = 0
		n2 = 0
		if comb_np[i][j] > 0:
			n1 = 1
		if comb_np[ii][jj] > 0:
			n2 = 1
		return n1 * n2 * interactions_list[comb_np[i][j]][comb_np[ii][jj]]

	#here we will calculate all interactions in one node (but it will be all interactions in the system because of overlaps)
	for comb in new_nodes_combinations:
		comb_np = np.full((triangular_number_order, triangular_number_order), None)
		counter = 0
		cur_mu = 0
		cur_en = 0
		for i in range(triangular_number_order):
			for j in range(i + 1):
				comb_np[i][j] = comb[counter]
				cur_mu += chem[comb[counter]] / nodes_for_chem_pot
				counter += 1
		combinations.append(comb_np)
		combinations_mu.append(cur_mu)
		
		#1NN
		dop_del = edges_with_bonds
		cur_en += add_en(1, 0, 1, 1, comb_np, interactions_list_1_right) / dop_del
		cur_en += add_en(2, 0, 2, 1, comb_np, interactions_list_1_right) / dop_del
		cur_en += add_en(2, 1, 2, 2, comb_np, interactions_list_1_right) / dop_del

		cur_en += add_en(0, 0, 1, 1, comb_np, interactions_list_1_diff) / dop_del
		cur_en += add_en(1, 1, 2, 2, comb_np, interactions_list_1_diff) / dop_del
		cur_en += add_en(1, 0, 2, 1, comb_np, interactions_list_1_diff) / dop_del

		cur_en += add_en(1, 0, 0, 0, comb_np, interactions_list_1_diff) / dop_del
		cur_en += add_en(2, 0, 1, 0, comb_np, interactions_list_1_diff) / dop_del
		cur_en += add_en(2, 1, 1, 1, comb_np, interactions_list_1_diff) / dop_del

		dop_del = 4
		#2NN
		cur_en += add_en(2, 0, 1, 1, comb_np, interactions_list_2_right) / dop_del
		cur_en += add_en(0, 0, 2, 2, comb_np, interactions_list_2_diff) / dop_del
		cur_en += add_en(2, 1, 0, 0, comb_np, interactions_list_2_diff) / dop_del

		
		#3NN
		cur_en += add_en(2, 0, 2, 2, comb_np, interactions_list_3_right) / dop_del
		cur_en += add_en(1, 0, 2, 2, comb_np, interactions_list_3_diff) / dop_del
		cur_en += add_en(2, 0, 0, 0, comb_np, interactions_list_3_diff) / dop_del

		combinations_en.append(cur_en)

	#here we should check nodes compatibility and calculate probability matrixes
	matrix_1 = []
	matrix_2 = []
	matrix_3 = []

	for n1, comb1 in enumerate(combinations):
		line_1 = []
		line_2 = []
		line_3 = []
		for n2, comb2 in enumerate(combinations):
			cur_mu_1 = combinations_mu[n1] + combinations_mu[n2]
			cur_mu_2 = combinations_mu[n1] + combinations_mu[n2]
			cur_mu_3 = 0#combinations_mu[n1] + combinations_mu[n2]

			#we should make some reductions in chemical potential because of overlaps of new nodes
			if triangular_number_order >= 2:
				cur_mu_1 -= chem[comb2[1][0]] / nodes_for_chem_pot
				cur_mu_2 -= chem[comb2[1][0]] / nodes_for_chem_pot
			if triangular_number_order >= 3:
				cur_mu_1 -= (chem[comb2[2][0]] + chem[comb2[2][1]])  / nodes_for_chem_pot
				cur_mu_2 -= (chem[comb2[2][0]] + chem[comb2[2][1]])  / nodes_for_chem_pot

			cur_en_1 = combinations_en[n1] + combinations_en[n2]
			cur_en_2 = combinations_en[n1] + combinations_en[n2]
			cur_en_3 = 0#combinations_en[n1] + combinations_en[n2]

			#we should make some reductions in energies because of overlaps of new nodes
			if triangular_number_order >= 3:
				dop_del = edges_with_bonds								
				cur_en_1 -= add_en(2, 0, 1, 0, comb2, interactions_list_1_diff) / dop_del								
				cur_en_1 -= add_en(1, 0, 2, 1, comb2, interactions_list_1_diff) / dop_del
				cur_en_1 -= add_en(2, 0, 2, 1, comb2, interactions_list_1_diff) / dop_del

				cur_en_2 -= add_en(2, 0, 1, 0, comb2, interactions_list_1_diff) / dop_del
				cur_en_2 -= add_en(1, 0, 2, 1, comb2, interactions_list_1_diff) / dop_del
				cur_en_2 -= add_en(2, 0, 2, 1, comb2, interactions_list_1_diff) / dop_del

			if comb1[0][0] == comb2[1][0] and comb1[1][1] == comb2[2][1] and comb1[1][0] == comb2[2][0]:
			#if comb1[0][0] == comb2[1][0]:
				line_1.append((cur_mu_1 - cur_en_1))
			else:
				line_1.append(inf)
			if comb1[1][1] == comb2[1][0] and comb1[2][1] == comb2[2][0] and comb1[2][2] == comb2[2][1]:
			#if comb1[1][1] == comb2[1][0]:
				line_2.append((cur_mu_2 - cur_en_2))
			else:
				line_2.append(inf)
			line_3.append(0)
		matrix_1.append(line_1)
		matrix_2.append(line_2)
		matrix_3.append(line_3)
	matrixes = [np.array(matrix_1) ,np.array(matrix_2) ,np.array(matrix_3)]
	return matrixes
