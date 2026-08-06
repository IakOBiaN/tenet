"""Lattice gases with interactions reaching beyond nearest neighbours."""
from itertools import product

import numpy as np

from tenet.models import inf, register


@register("1D_long-range", params = ["mu", "interactions", "mean_field"])
def one_dim_long_range(calc, temp, m_par):
	"""Chain with interactions out to the r-th neighbour.

	m_par: 0 - mu, 1 - list of interaction energies by neighbour distance,
	2 - mean field added to mu (used by the self-consistent driver).

	Merges as many sites as the interaction list is long and reports that
	through ``calc.nodes``.  The result is already a transfer matrix, so
	``simulate`` feeds it straight to ``tm_step`` without building a tensor.

	Drivers: one_dim_long_range.py, one_dim_long_range_with_mean_field.py.
	"""
	neigbours = calc.coord
	mu = m_par[0]
	mean_field = m_par[2]
	mu += mean_field
	chem = [0, mu]
	interactions_with_neighbours = m_par[1]

	merged_nodes = len(interactions_with_neighbours)
	calc.nodes = merged_nodes
	states = 2 #it is for two state models
	new_nodes_combinations = product(range(states), repeat = merged_nodes)
	combinations = []
	combinations_mu = []
	combinations_en = []
	for comb in new_nodes_combinations:
		cur_mu = chem[comb[0]]
		cur_en = 0
		for i in range(merged_nodes - 1):
			cur_mu += chem[comb[i + 1]]
			for j in range(len(interactions_with_neighbours)):
				inter_node = i + (j + 1)
				if inter_node >= merged_nodes:
					break
				cur_en += interactions_with_neighbours[j] * comb[i] * comb[i + (j + 1)]

		combinations.append(comb)
		combinations_mu.append(cur_mu / neigbours)
		combinations_en.append(cur_en / neigbours)

	#interactions between new nodes
	matrix = []
	for n1, comb1 in enumerate(combinations):
		line = []
		for n2, comb2 in enumerate(combinations):
			cur_mu = combinations_mu[n1] + combinations_mu[n2]
			cur_en = combinations_en[n1] + combinations_en[n2]
			for i in range(1, len(interactions_with_neighbours) + 1):
				for j in range(i):
					cur_en += interactions_with_neighbours[i - 1] * comb2[i - 1 - j] * comb1[-1 - j]
			line.append((cur_mu - cur_en))
		matrix.append(line)
	matrixes = [np.array(matrix) ,]
	return matrixes


@register("2D_long-range", params = ["mu", "interactions", "mean_field"])
def two_dim_long_range(calc, temp, m_par):
	"""Triangular lattice with interactions out to the third neighbour.

	m_par: 0 - mu, 1 - list of interaction energies by neighbour distance,
	2 - mean field added to mu.

	Sites are merged into triangular blocks, and the overlaps between blocks
	are then subtracted back out of both the chemical potential and the
	energy.  Sets ``calc.lattice`` to the square algorithm, because that is
	what the merged network is run through.

	Driver: 2D_long_range.py.
	"""
	mu = m_par[0]
	mean_field = m_par[2]
	mu += mean_field
	chem = [0, mu]
	interactions_with_neighbours = m_par[1]

	number_of_interactions = len(interactions_with_neighbours)
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

		for i in range(triangular_number_order):
			for j in range(i + 1):
				for ii in range(i, triangular_number_order):
					for jj in range(ii + 1):
						if i >= ii and j >= jj:
							continue
						neigb = pattern[pattern_center + ii - i][pattern_center + jj - j]
						dop_del = edges_with_bonds
						if neigb - 1 == 1:
							dop_del = 3 + 1
						if neigb - 1 == 2:
							dop_del = 3 + 1
						if neigb > 0 and neigb <= number_of_interactions:
							cur_en += comb_np[i][j] * comb_np[ii][jj] * interactions_with_neighbours[neigb - 1] / dop_del
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
			if triangular_number_order == 4:
				cur_mu_1 -= (chem[comb2[3][0]] + chem[comb2[3][1]] + chem[comb2[3][2]]) / nodes_for_chem_pot
				cur_mu_2 -= (chem[comb2[3][0]] + chem[comb2[3][1]] + chem[comb2[3][2]]) / nodes_for_chem_pot

			cur_en_1 = combinations_en[n1] + combinations_en[n2]
			cur_en_2 = combinations_en[n1] + combinations_en[n2]
			cur_en_3 = 0#combinations_en[n1] + combinations_en[n2]

			#we should make some reductions in energies because of overlaps of new nodes
			if triangular_number_order >= 3:
				cur_en_1 -= comb2[1][0] * comb2[2][0] * interactions_with_neighbours[0] / dop_del
				cur_en_1 -= comb2[1][0] * comb2[2][1] * interactions_with_neighbours[0] / dop_del
				cur_en_1 -= comb2[2][0] * comb2[2][1] * interactions_with_neighbours[0] / dop_del

				cur_en_2 -= comb2[1][0] * comb2[2][0] * interactions_with_neighbours[0] / dop_del
				cur_en_2 -= comb2[1][0] * comb2[2][1] * interactions_with_neighbours[0] / dop_del
				cur_en_2 -= comb2[2][0] * comb2[2][1] * interactions_with_neighbours[0] / dop_del

			if triangular_number_order == 4:
				#in future we should add elements here
				pass

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
