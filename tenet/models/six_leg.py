"""TPB + Cu monolayer on a six-leg tensor."""
from itertools import product

import numpy as np

from tenet.models import inf, register


@register("six_leg_test", params = ["mu_TPB", "mu_Cu", "w2", "w3", "w3_1", "w4", "eps"])
def six_leg_test(calc, temp, m_par):
	"""TPB + Cu co-adsorption, built directly as a rank-6 tensor.

	m_par: 0 - mu_TPB, 1 - mu_Cu, 2..5 - the two-, three- and four-particle
	interactions, 6 - eps (the close-contact penalty between TPB molecules).

	Each leg carries a pair of sub-states, so the tensor is 16**6 - about
	134 MB, which is why its golden case is marked slow.

	Driver: multy_test.py.
	"""
	#this model only appends its single tensor, so it needs its own list; in the
	#old if/elif chain it borrowed the one build_matrix declared before the chain
	matrixes = []

	mu_TPB = m_par[0]
	mu_Cu = m_par[1]

	w2 = -m_par[2]
	w3 = -m_par[3]
	w3_1 = - m_par[4]
	w4 = -m_par[5]
	eps = -m_par[6]

	#additions
	r_w2 = w2
	r_w3 = w3 - 2 * w2
	r_w3_1 = w3_1 - 2 * w2
	r_w4 = w4 - 3 * w2 - 3 * r_w3_1
	r_eps = eps

	states = tuple(product(range(4), repeat = 2))
	dimens_size = len(states)
	tensor = np.zeros((dimens_size, ) * 6)
	keys = {}
	for i, state in enumerate(states):
		keys[state] = i
	combination = product(states, repeat = 6)
	for state in combination:
		energy = 0
		if state[0][1] == state[1][1] == state[2][1] == state[3][0] == state[4][0] == state[5][0]:
			nodes = [state[0][1], state[0][0], state[1][0], state[2][0], state[3][1], state[4][1], state[5][1]]

			#close contact of TPB check
			close = [1 if (i == 1 or i == 2) else 0 for i in nodes]
			condition = 0
			#other neigbours
			for i in range(1, 7):
				next = i + 1
				if next > 6:
					next = 1
				condition += close[i] * close[next]
				condition += close[i] * close[0]
			if condition:
				tensor[keys[state[0]]][keys[state[1]]][keys[state[2]]][keys[state[3]]][keys[state[4]]][keys[state[5]]] = inf
				continue

			#chemical potential block
			sum_TPB = sum([1 for i in nodes if (i == 1 or i == 2)])
			sum_Cu = sum([1 for i in nodes if i == 3])
			energy += sum_TPB * mu_TPB / 7.0 + sum_Cu * mu_Cu / 7.0

			#if two TPB molecules too close, we add eps to the energy
			if close[1] == 1 and close[3] == 1:
				energy += r_eps
			if close[2] == 1 and close[4] == 1:
				energy += r_eps
			if close[3] == 1 and close[5] == 1:
				energy += r_eps

			sum_w2 = 0
			#pair interactions
			if nodes[1] == 2 and nodes[0] == 3:
				sum_w2 += 1
			if nodes[2] == 1 and nodes[0] == 3:
				sum_w2 += 1
			if nodes[3] == 2 and nodes[0] == 3:
				sum_w2 += 1
			if nodes[4] == 1 and nodes[0] == 3:
				sum_w2 += 1
			if nodes[5] == 2 and nodes[0] == 3:
				sum_w2 += 1
			if nodes[6] == 1 and nodes[0] == 3:
				sum_w2 += 1

			if nodes[1] == 3 and nodes[0] == 1:
				sum_w2 += 1
			if nodes[2] == 3 and nodes[0] == 2:
				sum_w2 += 1
			if nodes[3] == 3 and nodes[0] == 1:
				sum_w2 += 1
			if nodes[4] == 3 and nodes[0] == 2:
				sum_w2 += 1
			if nodes[5] == 3 and nodes[0] == 1:
				sum_w2 += 1
			if nodes[6] == 3 and nodes[0] == 2:
				sum_w2 += 1

			energy += sum_w2 * r_w2 / 2.0

			#triple interations line
			if nodes[2] == 1 and nodes[5] == 2 and nodes[0] == 3:
				energy += r_w3
			if nodes[1] == 2 and nodes[4] == 1 and nodes[0] == 3:
				energy += r_w3
			if nodes[3] == 2 and nodes[6] == 1 and nodes[0] == 3:
				energy += r_w3

			#triple interations angle
			if nodes[0] == 3 and nodes[1] == 2 and nodes[3] == 2:
				energy += r_w3_1
			if nodes[0] == 3 and nodes[2] == 1 and nodes[4] == 1:
				energy += r_w3_1
			if nodes[0] == 3 and nodes[3] == 2 and nodes[5] == 2:
				energy += r_w3_1
			if nodes[0] == 3 and nodes[4] == 1 and nodes[6] == 1:
				energy += r_w3_1
			if nodes[0] == 3 and nodes[5] == 2 and nodes[1] == 2:
				energy += r_w3_1
			if nodes[0] == 3 and nodes[6] == 1 and nodes[2] == 1:
				energy += r_w3_1

			#quad interactions
			if nodes[1] == 2 and nodes[3] == 2 and nodes[5] == 2 and nodes[0] == 3:
				energy += r_w4
			if nodes[2] == 1 and nodes[4] == 1 and nodes[6] == 1 and nodes[0] == 3:
				energy += r_w4

			tensor[keys[state[0]]][keys[state[1]]][keys[state[2]]][keys[state[3]]][keys[state[4]]][keys[state[5]]] = energy
		else:
			tensor[keys[state[0]]][keys[state[1]]][keys[state[2]]][keys[state[3]]][keys[state[4]]][keys[state[5]]] = inf
			continue
	matrixes.append(tensor)
	#print(count)
	#exit()
	return matrixes
