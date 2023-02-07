from math import exp, log,sqrt, pi, cos, radians
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm

inf = -1e8
constant = 1.

def build_matrix (model, temp, m_par, neigbours = 8.0):

	if len(m_par) < 10:
		m_par = m_par + [0.0]*(10-len(m_par))

	models_dict = {
		"langmuir" : True,
		"ising" : True,
		"hard-disk" : True,
		"dimers" : True,
		"1NN" : True,
		"2NN" : True,
		"3NN" : True,
		"4NN" : True,
		"5NN" : True,
		"HT1" : True,
		"HT2" : True,
		"HT3" : True,
		"qstate" : True
	}

	exist = models_dict.get(model)
	assert (exist is not None), "Error! This model is not in the database"

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrixes = []
	if model == "langmuir":
		matrixes = [np.array([[0.0, m_par[0]/neigbours],[m_par[0]/neigbours, -m_par[1]+m_par[0]/(neigbours/2.0)]]) ,] * 3
	elif model == "ising":
		matrixes = [np.array([[(m_par[1]-m_par[0]/(neigbours/2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1]+m_par[0]/(neigbours/2.0))]]), ] * 3
	elif model == "hard-disk":
		matrixes = [np.array([[0.0, m_par[0]/(neigbours/1.0)],[m_par[0]/(neigbours/1.0), -1000000.0+m_par[0]]]), ] * 3
	elif model == "dimers":
		matrixes = [np.array([[0, inf, (m_par[0]+m_par[1])/6.0, (m_par[0]+m_par[1])/6.0, m_par[0]/3.0], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[inf, (m_par[0]+m_par[1])/3.0, inf, inf, inf], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[m_par[0]/3.0, inf, inf, inf, inf]]), \
					np.array([[0, (m_par[0]+m_par[1])/6.0, (m_par[0]+m_par[1])/6.0, inf, m_par[0]/3.0], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[inf, inf, inf, (m_par[0]+m_par[1])/3.0, inf], \
						[m_par[0]/3.0, inf, inf, inf, inf]])]
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
	elif model == "HT1":
		matrixes = [np.array([[0, m_par[0], m_par[0], 2.0 * m_par[0] + m_par[4]], \
						[m_par[0], 2.0 * m_par[0], 2.0 * m_par[0], 3.0 * m_par[0] + m_par[4]], \
						[m_par[0], 2.0 * m_par[0] + m_par[1], 2.0 * m_par[0], 3.0 * m_par[0] + m_par[4] + m_par[1]], \
						[2.0 * m_par[0] + m_par[4], 3.0 * m_par[0] + m_par[4] + m_par[1], 3.0 * m_par[0] + m_par[4], 4.0 * m_par[0] + 2.0 * m_par[4] + m_par[1]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0 * m_par[0] + m_par[4]], \
						[m_par[0], 2.0 * m_par[0] + m_par[2], 2.0 * m_par[0], 3.0 * m_par[0] + m_par[4] + m_par[2]], \
						[m_par[0], 2.0 * m_par[0] + m_par[4], 2.0 * m_par[0] + m_par[2], 3.0 * m_par[0] + 2.0 * m_par[4] + m_par[2]], \
						[2.0 * m_par[0] + m_par[4], 3.0 * m_par[0] + 2.0 * m_par[4] + m_par[2], 3.0 * m_par[0] + m_par[4] + m_par[2], 4.0 * m_par[0] + 3.0 * m_par[4] + 2.0 * m_par[2]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0 * m_par[0] + m_par[4]], \
						[m_par[0], 2.0 * m_par[0] + m_par[2], 2.0 * m_par[0] + m_par[1], 3.0 * m_par[0] + m_par[4] + m_par[1] + m_par[2]], \
						[m_par[0], 2.0*m_par[0] + m_par[1], 2.0 * m_par[0] + m_par[2], 3.0 * m_par[0] + m_par[4] + m_par[1] + m_par[2]], \
						[2.0 * m_par[0] + m_par[4], 3.0 * m_par[0] + m_par[4] + m_par[1] + m_par[2], 3.0 * m_par[0] + m_par[4] + m_par[1] + m_par[2], 4.0 * m_par[0] + 2.0 * m_par[4] + 2.0 * m_par[1] + 2.0 * m_par[2]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0 * m_par[0] + m_par[4]], \
						[m_par[0], 2.0 * m_par[0] + m_par[2], 2.0 * m_par[0] + m_par[4], 3.0 * m_par[0] + 2.0 * m_par[4] + m_par[2]], \
						[m_par[0], 2.0 * m_par[0], 2.0 * m_par[0] + m_par[2], 3.0 * m_par[0] + m_par[4] + m_par[2]], \
						[2.0 * m_par[0] + m_par[4], 3.0 * m_par[0] + m_par[4] + m_par[2], 3.0 * m_par[0] + 2.0 * m_par[4] + m_par[2], 4.0 * m_par[0] + 3.0 * m_par[4] + 2.0 * m_par[2]]])]
	elif model == "HT2":
		matrixes = [np.array([[0, 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0 + m_par[5], m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[5], 2.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, m_par[0] / 4.0], \
						[2.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[3], inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[5], 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0], \
						[2.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[4], inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[5], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4], inf, 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[4]], \
						[2.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[5] + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[5]], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[5], 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[5], 2.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0], \
						[2.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[5], 4.0 * m_par[0] / 4.0 + m_par[5] + m_par[4], inf, 3.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[5]], \
						[2.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4], 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[4]], \
						[2.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[1], 3.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[3], 3.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, inf, inf, inf, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[4], inf, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, inf, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0]]), \
					np.array([[0, 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, inf, m_par[0] / 4.0, m_par[0] / 4.0], \
						[inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[4], inf, inf, inf], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[3], inf, inf, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[4], inf, inf], \
						[2.0 * m_par[0] / 4.0 + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[5] + m_par[4] + m_par[3], inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[5], inf], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[2], 2.0 * m_par[0] / 4.0 + m_par[3], 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[3], 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[5], inf, inf, inf, inf, 2.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[5], inf, inf], \
						[2.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4], 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[5], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4] + m_par[5], 4.0 * m_par[0] / 4.0 + m_par[4] + 2.0 * m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[3], inf], \
						[2.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4], 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[2], 3.0 * m_par[0] / 4.0 + m_par[3]], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0, inf, inf], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, 2.0 * m_par[0] / 4.0 + m_par[3], inf]]), \
					np.array([[0, 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, inf], \
						[2.0 * m_par[0] / 4.0, inf, 3.0 * m_par[0] / 4.0 + m_par[3], 4.0 * m_par[0] / 4.0 + m_par[3] + m_par[4], inf, inf, inf, inf, 4.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, inf], \
						[inf, inf, inf, inf, inf, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[4], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4], inf, inf, inf], \
						[inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[5] + m_par[2], inf, inf, inf, inf], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[3], 2.0 * m_par[0] / 4.0 + m_par[2], 3.0 * m_par[0] / 4.0 + m_par[2] + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[2] + m_par[5], inf, inf, inf, 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[1], inf], \
						[inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[2] + 2.0 * m_par[5], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[5]], \
						[2.0 * m_par[0] / 4.0 + m_par[4], 4.0 * m_par[0] / 4.0 + m_par[3] + m_par[4], 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[4], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4] + 2.0 * m_par[1] + 2.0 * m_par[2], inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[3], 3.0 * m_par[0] / 4.0 + m_par[1] + m_par[2], inf], \
						[2.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[3], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[3], inf], \
						[inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, inf, inf, 2.0 * m_par[0] / 4.0], \
						[m_par[0] / 4.0, 3.0 * m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[1], 3.0 * m_par[0] / 4.0 + m_par[1] + m_par[2], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[3], 2.0 * m_par[0] / 4.0 + m_par[2], inf]]), \
					np.array([[0, inf, m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[4], inf, m_par[0] / 4.0, 2.0 * m_par[0] / 4.0 + m_par[5], inf, 2.0 * m_par[0] / 4.0, m_par[0] / 4.0, inf], \
						[2.0 * m_par[0] / 4.0, inf, 3.0 * m_par[0] / 4.0 + m_par[2], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4] + m_par[2], inf, 3.0 * m_par[0] / 4.0 + m_par[3], 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[5] + m_par[3], inf, 4.0 * m_par[0] / 4.0 + m_par[5], 3.0 * m_par[0] / 4.0 + m_par[4], inf], \
						[inf, 3.0 * m_par[0] / 4.0, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, 4.0 * m_par[0] / 4.0 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, 4.0 * m_par[0] / 4.0 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0] / 4.0, inf, 2.0 * m_par[0] / 4.0 + m_par[3], 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[3] + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[4], 2.0 * m_par[0] / 4.0 + m_par[5], inf], \
						[2.0 * m_par[0] / 4.0, inf, 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[4] + m_par[3] + m_par[5], inf, inf], \
						[2.0 * m_par[0] / 4.0 + m_par[4], inf, 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[3], 4.0 * m_par[0] / 4.0 + 2.0 * m_par[3] + 2.0 * m_par[4], inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + 2.0 * m_par[4] + m_par[2], 3.0 * m_par[0] / 4.0 + m_par[4] + m_par[5] + m_par[3], inf], \
						[inf, inf, inf, inf, 4.0 * m_par[0] / 4.0 + m_par[5], inf, inf, 4.0 * m_par[0] / 4.0, inf, inf, 3.0 * m_par[0] / 4.0], \
						[m_par[0] / 4.0, inf, 2.0 * m_par[0] / 4.0, inf, inf, 2.0 * m_par[0] / 4.0, inf, inf, 3.0 * m_par[0] / 4.0 + m_par[3], inf, inf], \
						[m_par[0] / 4.0, inf, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[4], inf, 2.0 * m_par[0] / 4.0, 3.0 * m_par[0] / 4.0 + m_par[3] + m_par[5], inf, 3.0 * m_par[0] / 4.0 + m_par[2], 2.0 * m_par[0] / 4.0 + m_par[3], inf]])]
	elif model == "HT3":
		matrixes = [np.array([[0, m_par[0] / 9, m_par[0] / 9, m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[5], 2.0 * m_par[0] / 9 + m_par[5], 2.0 * m_par[0] / 9 + m_par[4], 2.0 * m_par[0] / 9, inf, inf, inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[4], 2.0 * m_par[0] / 9 + m_par[5], 2.0 * m_par[0] / 9 + m_par[5], 2.0 * m_par[0] / 9, inf, 2.0 * m_par[0] / 9], \
						[m_par[0] / 9, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[4], 3.0 * m_par[0] / 9 + m_par[5], inf, inf, 3.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[3], inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, 4.0 * m_par[0] / 9 + m_par[4]], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[2], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 3.0 * m_par[0] / 9, inf, 3.0 * m_par[0] / 9], \
						[2.0 * m_par[0] / 9 + m_par[5], inf, inf, 3.0 * m_par[0] / 9 + m_par[5] + m_par[3], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[5], inf], \
						[2.0 * m_par[0] / 9 + m_par[4], 3.0 * m_par[0] / 9 + m_par[4] + m_par[3], inf, inf, 4.0 * m_par[0] / 9 + m_par[4] + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[4] + m_par[5] + m_par[3], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[5] + 2.0 * m_par[3], inf, inf, inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, 4.0 * m_par[0] / 9 + m_par[5]], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf, 3.0 * m_par[0] / 9], \
						[2.0 * m_par[0] / 9 + m_par[5], inf, inf, 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf], \
						[2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[2], 2.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 3.0 * m_par[0] / 9 + m_par[4] + m_par[3], 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf, 3.0 * m_par[0] / 9], \
						[2.0 * m_par[0] / 9 + m_par[4], inf, inf, 3.0 * m_par[0] / 9 + m_par[3] + m_par[4], 4.0 * m_par[0] / 9 + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[3], 3.0 * m_par[0] / 9 + m_par[4] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + 2.0 * m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf], \
						[2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[2], 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[1], 4.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + m_par[5] + m_par[4], 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[5] + m_par[4], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[5], 4.0 * m_par[0] / 9 + m_par[3], inf, 4.0 * m_par[0] / 9 + m_par[3]], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[2], 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf]]), \
					np.array([[0, m_par[0] / 9, m_par[0] / 9, m_par[0] / 9, 2.0 * m_par[0] / 9, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[4], 2.0 * m_par[0] / 9 + m_par[5], 2.0 * m_par[0] / 9 + m_par[5], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[3], inf, inf, 3.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[2], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5]], \
						[2.0 * m_par[0] / 9 + m_par[4], 3.0 * m_par[0] / 9 + m_par[4], inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + 2.0 * m_par[5] + 2.0 * m_par[3], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4]], \
						[2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5]], \
						[2.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[2], 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf]]), \
					np.array([[0, m_par[0] / 9, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[4], inf, inf, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, inf], \
						[inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf], \
						[inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[2], inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[1], 3.0 * m_par[0] / 9 + m_par[1] + m_par[2] + m_par[4], inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3], inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + m_par[5], inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[4], inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[1], inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[2], 3.0 * m_par[0] / 9 + m_par[1] + m_par[2] + m_par[4], inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[5], inf], \
						[inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + 2.0 * m_par[5] + m_par[2], inf, inf, inf, inf], \
						[inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[2] + 2.0 * m_par[5], inf, inf, inf], \
						[2.0 * m_par[0] / 9 + m_par[4], 3.0 * m_par[0] / 9 + m_par[1] + m_par[2], inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[1] + m_par[2] + m_par[4], 4.0 * m_par[0] / 9 + 2.0 * m_par[1] + 2.0 * m_par[2] + 2.0 * m_par[4], inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf], \
						[inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[2] + 2.0 * m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + 2.0 * m_par[2] + 2.0 * m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3], inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf]]), \
					np.array([[0, m_par[0] / 9, inf, inf, inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[5], inf, inf, m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[4], inf, inf, 2.0 * m_par[0] / 9, 2.0 * m_par[0] / 9, inf], \
						[m_par[0] / 9, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf], \
						[m_par[0] / 9, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, 2.0 * m_par[0] / 9, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[3], inf, inf, inf, inf, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, inf, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, 2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[3] + m_par[4], inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[2], inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, 3.0 * m_par[0] / 9 + m_par[2], 4.0 * m_par[0] / 9 + 2.0 * m_par[4] + m_par[2], inf, inf, 4.0 * m_par[0] / 9 + m_par[5], 4.0 * m_par[0] / 9 + m_par[5], inf], \
						[2.0 * m_par[0] / 9, 3.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3], 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[4], 4.0 * m_par[0] / 9 + m_par[5], inf], \
						[m_par[0] / 9, 2.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0 * m_par[0] / 9 + m_par[3], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, 3.0 * m_par[0] / 9 + m_par[5], 3.0 * m_par[0] / 9 + m_par[4], inf], \
						[2.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf], \
						[2.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[3] + m_par[5], inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf], \
						[2.0 * m_par[0] / 9 + m_par[4], 3.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9 + m_par[4] + m_par[3], 4.0 * m_par[0] / 9 + 2.0 * m_par[3] + 2.0 * m_par[4] + m_par[5], inf, inf, 4.0 * m_par[0] / 9 + m_par[3] + m_par[4] + m_par[5], 4.0 * m_par[0] / 9 + m_par[2] + 2.0 * m_par[4], inf], \
						[inf, inf, inf, inf, 4.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0 * m_par[0] / 9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, 3.0 * m_par[0] / 9, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9], \
						[inf, inf, inf, 3.0 * m_par[0] / 9, inf, inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[4], inf, inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf, 4.0 * m_par[0] / 9 + m_par[5], inf, inf, inf]])]
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
		#print(matrixes)
		#exit()

	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i]/(constant*temp)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])
	return matrixes

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

	U, S, V = split_by_svd(tensor, [0, 1], [2, 3])
	S = np.sqrt(S)
	U = np.einsum("abi,i->abi", U, S)
	V = np.einsum("ibc,i->ibc",V, S)
	tensor_1 = np.tensordot(id3, V, ([1], [2]))
	tensor_2 = np.tensordot(U, id3, ([1], [1]))
	tensor = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
	tensor = np.swapaxes(tensor, 2, 3)
	tensor = list((tensor, ))

	"""one = np.einsum("ab,iak->ibk",one,cd)
	two = np.einsum("ab,iak->ibk",three,cd)
	three = np.einsum("ab,ibk->iak",two2,cd)
	tensor = np.einsum("abc,deb,ice->adi",one,two,three)
	tensor = np.einsum("abi,ijk->abjk", tensor,cd)
	tensor = list((tensor, ))"""
	return tensor
