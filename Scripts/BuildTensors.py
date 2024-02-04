from math import exp, log,sqrt, pi, cos, radians
import numpy as np
from itertools import product
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm

inf = -1e8

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
		"Pentacene_model_1" : True,
		"CHD_complex" : True
	}

	exist = models_dict.get(calc.model)
	assert (exist is not None), "Error! This model is not in the database"

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrixes = []
	if model == "langmuir":
		matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3
	#m_par: 0 - mu, 1 - eps, 2 - multipartical interaction
	if model == "langmuir_m":
		mult = np.zeros((2, 2, 2))
		mult[1, 1, 1] = -m_par[2]
		matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3 + [mult]
	elif model == "binary":
		#m_par: 0 - muA, 1 - muB, 2 - epsAA, 3 - epsBB, 4 - epsAB
		matrixes = [np.array([[0.0, m_par[0] / neigbours, m_par[1] / neigbours], [m_par[0] / neigbours, -m_par[2] + 2.0 * m_par[0] / neigbours, (m_par[0] + m_par[1]) / neigbours], [m_par[1] / neigbours, (m_par[0] + m_par[1]) / neigbours, -m_par[3] + 2.0 * m_par[1] / neigbours]]) ,] * 3
	elif model == "ising":
		matrixes = [np.array([[(m_par[1] - m_par[0] / (neigbours / 2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1] + m_par[0] / (neigbours / 2.0))]]), ] * 3
	elif model == "hard-disk":
		matrixes = [np.array([[0.0, m_par[0] / (neigbours)],[m_par[0] / (neigbours), inf + m_par[0]]]), ] * 3
	elif model == "TLAT":
		matrixes = [np.array([[-m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3], m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3]], [-m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3], m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3]], [m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3], -m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3]], [m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3], -m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3]]]), ] * 3
	elif model == "Pentacene_model_1":
		mu = m_par[0] / neigbours
		e_close = -m_par[1]
		e_one = -m_par[2]
		e_two = -m_par[3]
		states = 3
		nodes = 4
		calc.nodes = nodes
		"""exist = [[1, 1, 0, 0, 0], \
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
		chem = [0, mu / 4.0, mu / 4.0, mu / 4.0, mu / 4.0]"""
		exist = [[1, 1, 0], \
						[0, 0, 1], \
						[1, 1, 0]]
		energies = [[0, 0, 0], \
						[0, 0, 0], \
						[0, e_close, 0]]
		#combination with e_one energy
		energy_one = [2, 0, 1]
		#combination with e_two energy
		energy_two = [2, 0, 0, 1]
		chem = [0, mu / 2.0, mu / 2.0]
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
			if len(cur) > 2:
				for i in range(len(cur) - 2):
					if list(cur[i:i + 3]) == list(energy_one):
						cur_en += e_one / 2.0
			if len(cur) > 3:
				for i in range(len(cur) - 3):
					if list(cur[i:i + 4]) == list(energy_two):
						cur_en += e_two / 2.0
			if comb_no:
				continue
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
				cur = [1,2,9,8]
				for i in range(2):
					if i + 3 > len(cur):
						break
					comp_list_one = list(cur[i:i + 3])
					if comp_list_one == energy_one:
						cur_en += e_one
				for i in range(3):
					if i + 4 > len(cur):
						break
					comp_list_two = list(cur[i:i + 4])
					if comp_list_two == energy_two:
						cur_en += e_two
				line.append(cur_mu + cur_en)
				line2.append(cur_mu)
			mat1.append(line)
			mat2.append(line2)
		matrixes = [np.array(mat1), np.array(mat2)]
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
