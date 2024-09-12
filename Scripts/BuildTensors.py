from math import exp, log,sqrt, pi, cos, radians
import numpy as np
from itertools import product
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm

inf = -1e8

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def build_matrix (calc, temp, m_par):

	model = calc.model
	neigbours = calc.coord

	if len(m_par) < 10:
		m_par = m_par + [0.0] * (10 - len(m_par))

	models_dict = {
		"langmuir" : True,
		"langmuir_m" : True,
		"binary" : True,
		"ising" : True,
		"hard-disk" : True,
		"TLAT" : True,
		"dimers" : True,
		"1NN" : True,
		"2NN" : True,
		"3NN" : True,
		"4NN" : True,
		"5NN" : True,
		"HT1" : True,
		"HT2" : True,
		"HT3" : True,
		"qstate" : True,
		"CHD_simple" : True,
		"Pentacene_model_1_simple" : True,
		"Pentacene_model_1_complex" : True,
		"Pentacene_model_2" : True,
		"Pentacene_model_3" : True,
		"CHD_complex" : True,
		"six_leg_test" : True,
		"dimers_test" : True
	}

	exist = models_dict.get(calc.model)
	assert (exist is not None), "Error! This model is not in the database"

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrixes = []
	if model == "langmuir":
		matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3
	#m_par: 0 - mu, 1 - eps, 2 - multipartical interaction
	elif model == "langmuir_m":
		mult = np.zeros((2, 2, 2))
		mult[1, 1, 1] = -m_par[2]
		matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3 + [mult]
	elif model == "six_leg_test":
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
	elif model == "binary":
		#m_par: 0 - muA, 1 - muB, 2 - epsAA, 3 - epsBB, 4 - epsAB
		matrixes = [np.array([[0.0, m_par[0] / neigbours, m_par[1] / neigbours], [m_par[0] / neigbours, -m_par[2] + 2.0 * m_par[0] / neigbours, (m_par[0] + m_par[1]) / neigbours], [m_par[1] / neigbours, (m_par[0] + m_par[1]) / neigbours, -m_par[3] + 2.0 * m_par[1] / neigbours]]) ,] * 3
	elif model == "ising":
		matrixes = [np.array([[(m_par[1] - m_par[0] / (neigbours / 2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1] + m_par[0] / (neigbours / 2.0))]]), ] * 3
	elif model == "hard-disk":
		matrixes = [np.array([[0.0, m_par[0] / (neigbours)],[m_par[0] / (neigbours), inf + m_par[0]]]), ] * 3
	elif model == "TLAT":
		matrixes = [np.array([[-m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3], m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3]], [-m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3], m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3]], [m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3], -m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3]], [m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3], -m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3]]]), ] * 3
	elif model == "Pentacene_model_1_simple":
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
	elif model == "Pentacene_model_1_complex":
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
	elif model == "Pentacene_model_2":
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
	elif model == "Pentacene_model_3":
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

	elif model == "CHD_simple":
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
	elif model == "CHD_complex":
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
	#matrixes only for hex lattice
	elif model == "dimers":
		neigbours = 3.0
		calc.nodes = 2.0
		matrixes = [np.array([[0, inf, (m_par[0] + m_par[1]) / (neigbours * 2.0), (m_par[0] + m_par[1]) / (neigbours * 2.0), m_par[0] / neigbours], \
						[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
						[inf, (m_par[0] + m_par[1]) / neigbours, inf, inf, inf], \
						[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
						[m_par[0] / neigbours, inf, inf, inf, inf]]), \
					np.array([[0, (m_par[0] + m_par[1]) / (neigbours * 2.0), (m_par[0] + m_par[1]) / (neigbours * 2.0), inf, m_par[0] / neigbours], \
						[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
						[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
						[inf, inf, inf, (m_par[0] + m_par[1]) / neigbours, inf], \
						[m_par[0] / neigbours, inf, inf, inf, inf]])]
	elif model == "dimers_test":
		mu = m_par[0] / neigbours
		matrixes = [np.array([[0, mu / 2.0, inf, mu / 2.0, mu / 2.0, mu], \
						[inf, inf, mu, inf, inf, inf], \
						[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
						[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
						[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
						[mu, 3.0 / 2.0 * mu, inf, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, 2.0 * mu]]), \
					np.array([[0, mu / 2.0, mu / 2.0, mu / 2.0, inf, mu], \
						[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
						[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
						[inf, inf, inf, inf, mu, inf], \
						[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
						[mu, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, inf, 2.0 * mu]])]
	elif model == "1NN" or model == "2NN" or model == "3NN" or model == "4NN" or model == "5NN":
		var_1NN_0 = 0
		var_1NN_mu = m_par[0] / neigbours
		var_2NN_0 = 0
		var_2NN_mu = m_par[0] / neigbours
		var_3NN = 0
		var_4NN = 0
		if model == "1NN":
			var_1NN_0 = 0
			var_1NN_mu = m_par[0] / neigbours
			var_2NN_0 = 0
			var_2NN_mu = m_par[0] / neigbours
			var_3NN = 0
			var_4NN = 0
		elif model == "2NN":
			var_1NN_0 = inf
			var_1NN_mu = inf
			var_2NN_0 = 0
			var_2NN_mu = m_par[0] / neigbours
			var_3NN = 0
			var_4NN = 0
		elif model == "3NN":
			var_1NN_0 = inf
			var_1NN_mu = inf
			var_2NN_0 = inf
			var_2NN_mu = inf
			var_3NN = 0
			var_4NN = 0
		elif model == "4NN":
			var_1NN_0 = inf
			var_1NN_mu = inf
			var_2NN_0 = inf
			var_2NN_mu = inf
			var_3NN = inf
			var_4NN = 0
		elif model == "5NN":
			var_1NN_0 = inf
			var_1NN_mu = inf
			var_2NN_0 = inf
			var_2NN_mu = inf
			var_3NN = inf
			var_4NN = inf
		matrixes = [np.array([[0, inf, 0, inf, inf, inf, 0, 0], \
						[inf, inf, var_1NN_mu, inf, m_par[0] / neigbours, inf, var_1NN_mu, var_2NN_mu], \
						[inf, inf, inf, 0, inf, var_1NN_0, var_2NN_0, var_1NN_0], \
						[0, var_1NN_mu, var_2NN_0, inf, inf, var_2NN_0, var_3NN, var_3NN], \
						[0, var_2NN_mu, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN, var_4NN], \
						[0, var_1NN_mu, var_3NN, var_2NN_0, inf, inf, var_2NN_0, var_3NN], \
						[inf, inf, var_2NN_0, var_1NN_0, inf, 0, inf, var_1NN_0], \
						[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf]]), \
					np.array([[0, inf, 0, 0, inf, inf, inf, 0], \
						[inf, inf, var_2NN_mu, var_1NN_mu, inf, m_par[0] / neigbours, inf, var_1NN_mu], \
						[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf], \
						[inf, inf, var_1NN_0, inf, 0, inf, var_1NN_0, var_2NN_0], \
						[0, var_1NN_mu, var_3NN, var_2NN_0, inf, var_1NN_0, var_2NN_0, var_3NN], \
						[0, var_2NN_mu, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN], \
						[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
						[inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0, inf]]), \
					np.array([[0, inf, 0, 0, 0, inf, inf, inf], \
						[inf, inf, var_1NN_mu, var_2NN_mu, var_1NN_mu, inf, m_par[0] / neigbours, inf], \
						[inf, inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0], \
						[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf], \
						[inf, inf, var_2NN_0, var_1NN_0, inf, 0, inf, var_1NN_0], \
						[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
						[0, var_2NN_mu, var_3NN, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0], \
						[0, var_1NN_mu, var_2NN_0, var_3NN, var_3NN, var_2NN_0, inf, inf]])]
	elif model == "qstate":
		mu = m_par[0] / 6
		c = m_par[1]
		n = m_par[2]
		epsilon = m_par[3]
		delta = m_par[4]
		matrixes = []
		#right-up
		anglesi = [(i - radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		anglesj = [(i + pi - radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		matrix = [[0, ] + [mu, ] * n]
		for alpha_i in anglesi:
			line = [mu, ]
			for alpha_j in anglesj:
				uij = 0
				for k in range(c):
					for l in range (c):
						if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
							uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
				uij += delta
				line.append(-uij + 2.0 * mu)
			matrix.append(line)
		matrixes.append(np.array(matrix))
		#right
		anglesi = [i for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		anglesj = [(i - pi) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		matrix = [[0, ] + [mu, ] * n]
		for alpha_i in anglesi:
			line = [mu, ]
			for alpha_j in anglesj:
				uij = 0
				for k in range(c):
					for l in range (c):
						if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
							uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
				uij += delta
				line.append(-uij + 2.0 * mu)
			matrix.append(line)
		matrixes.append(np.array(matrix))
		#right-bottom
		anglesi = [(i + radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		anglesj = [(i + pi + radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
		matrix = [[0, ] + [mu, ] * n]
		for alpha_i in anglesi:
			line = [mu, ]
			for alpha_j in anglesj:
				uij = 0
				for k in range(c):
					for l in range (c):
						if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
							uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
				uij += delta
				line.append(-uij + 2.0 * mu)
			matrix.append(line)
		matrixes.append(np.array(matrix))

	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i] / (calc.constant * temp)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])
	return matrixes

#old code for triangles
"""
def build_triangles_tensor(model, temp, m_par):
	inf = -1e8
	if model == "hard_triangles":
		field = m_par[0]/6.0
		one = np.array([[0, inf, field, field, field, inf, field], \
						[field, inf, 2.0*field, 2.0*field, 2.0*field, inf, 2.0*field], \
						[inf, 2.0*field, inf, inf, inf, inf, inf], \
						[field, inf, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[field, inf, 2.0*field, 2.0*field, inf, inf, 2.0*field], \
						[field, inf, 2.0*field, 2.0*field, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf]])
		two = np.array([[0, field, field, inf, field, inf, field], \
						[field, inf, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf]])
		three = np.array([[0, field, field, inf, field, field, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, inf, inf, 2.0*field, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, 2.0*field, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, 2.0*field, inf]])
	elif model == "hard_triangles2":
		field = m_par[0]/6.0
		one = np.array([[0, inf, inf, inf, field, field, field, field, inf, inf, field, inf, field], \
						[field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, 2.0*field], \
						[field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, 2.0*field], \
						[field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf]])
		two = np.array([[0, field, field, inf, field, inf, inf, field, inf, inf, field, inf, field], \
						[field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, inf, inf]])
		three = np.array([[0, field, field, inf, field, inf, inf, field, field, field, inf, inf, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf]])
	elif model == "hard_triangles3":
		field = m_par[0]/6.0
		one = np.array([[0, inf, inf, inf, inf, inf, inf, field, field, field, field, field, inf, inf, inf, field, inf, inf, field, inf, field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf]])
		two = np.array([[0, field, field, inf, field, inf, inf, field, inf, inf, inf, field, inf, inf, inf, field, inf, inf, field, inf, field], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf]])
		three = np.array([[0, field, field, inf, field, inf, inf, field, inf, inf, inf, field, field, field, field, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf, 2.0*field, inf, inf, inf, 2.0*field, 2.0*field, 2.0*field, 2.0*field, inf, inf, inf, inf, inf, inf]])

	one *= 1./(constant*temp)
	one = np.array([np.exp(line) for line in one])
	two *= 1./(constant*temp)
	two = np.array([np.exp(line) for line in two])
	two2 = two
	three *= 1./(constant*temp)
	three = np.array([np.exp(line) for line in three])
	cd = identity(3, one.shape[0])

	id4 = identity(4, one.shape[0])
	id3 = identity(3, one.shape[0])
	#dop = three
	#three = two
	#two = dop
	one = one ** 0.5
	two = two ** 0.5
	tensor = np.einsum("abcd, bi, nd, cg -> aign", id4, two, one, three)
	tensor = np.einsum("ijkl, ajc, xc -> iaxkl", tensor, id3, one)
	tensor = np.einsum("ijklm, ablk, xb -> ijaxm", tensor, id4, two)
	tensor = np.einsum("ijklm, aml -> ijka", tensor, id3)

	#tensor = np.einsum("abcd, efgh, ijk, mno, bi, fk, nd, oh, cg -> ajem", tensor, id4, id3, id3, two, one, one, two, three)

	U, S, V = tensor_svd(tensor, [0, 1], [2, 3])
	S = np.sqrt(S)
	U = np.einsum("abi,i->abi", U, S)
	V = np.einsum("ibc,i->ibc",V, S)
	tensor_1 = np.tensordot(id3, V, ([1], [2]))
	tensor_2 = np.tensordot(U, id3, ([1], [1]))
	tensor = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
	tensor = np.swapaxes(tensor, 2, 3)
	tensor = list((tensor, ))

	#one = np.einsum("ab,iak->ibk",one,cd)
	#two = np.einsum("ab,iak->ibk",three,cd)
	#three = np.einsum("ab,ibk->iak",two2,cd)
	#tensor = np.einsum("abc,deb,ice->adi",one,two,three)
	#tensor = np.einsum("abi,ijk->abjk", tensor,cd)
	#tensor = list((tensor, ))
	return tensor"""
