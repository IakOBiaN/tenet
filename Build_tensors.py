from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm

inf = -1e8
constant = 1.

def build_matrix (model, temp, m_par, neigbours = 8.0):

	if len(m_par) < 10:
		m_par = m_par + [0.0]*(10-len(m_par))


	var_0NN = 0
	var_1NN_0 = 0
	var_1NN_mu = m_par[0]/neigbours
	var_2NN_0 = 0
	var_2NN_mu = m_par[0]/neigbours
	var_3NN = 0
	var_4NN = 0
	NN_model = "3NN"
	if NN_model == "1NN":
		var_0NN = inf
		var_1NN_0 = 0
		var_1NN_mu = m_par[0]/neigbours
		var_2NN_0 = 0
		var_2NN_mu = m_par[0]/neigbours
		var_3NN = 0
		var_4NN = 0
	elif NN_model == "2NN":
		var_0NN = inf
		var_1NN_0 = inf
		var_1NN_mu = inf
		var_2NN_0 = 0
		var_2NN_mu = m_par[0]/neigbours
		var_3NN = 0
		var_4NN = 0
	elif NN_model == "3NN":
		var_0NN = inf
		var_1NN_0 = inf
		var_1NN_mu = inf
		var_2NN_0 = inf
		var_2NN_mu = inf
		var_3NN = 0
		var_4NN = 0
	elif NN_model == "4NN":
		var_0NN = inf
		var_1NN_0 = inf
		var_1NN_mu = inf
		var_2NN_0 = inf
		var_2NN_mu = inf
		var_3NN = inf
		var_4NN = 0
	elif NN_model == "5NN":
		var_0NN = inf
		var_1NN_0 = inf
		var_1NN_mu = inf
		var_2NN_0 = inf
		var_2NN_mu = inf
		var_3NN = inf
		var_4NN = inf

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrix_dict = {
		"langmuir" : (np.array([[0.0, m_par[0]/neigbours],[m_par[0]/neigbours, -m_par[1]+m_par[0]/(neigbours/2.0)]]) ,) * 3,
		"ising" : (np.array([[(m_par[1]-m_par[0]/(neigbours/2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1]+m_par[0]/(neigbours/2.0))]]), ) * 3,
		"hard-disk" : (np.array([[0.0, m_par[0]/(neigbours/1.0)],[m_par[0]/(neigbours/1.0), -1000000.0+m_par[0]]]), ) * 3,
		"dimers" : (np.array([[0, inf, (m_par[0]+m_par[1])/6.0, (m_par[0]+m_par[1])/6.0, m_par[0]/3.0], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[inf, (m_par[0]+m_par[1])/3.0, inf, inf, inf], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[m_par[0]/3.0, inf, inf, inf, inf]]), \
					np.array([[0, (m_par[0]+m_par[1])/6.0, (m_par[0]+m_par[1])/6.0, inf, m_par[0]/3.0], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[(m_par[0]+m_par[1])/6.0, inf, inf, inf, inf], \
						[inf, inf, inf, (m_par[0]+m_par[1])/3.0, inf], \
						[m_par[0]/3.0, inf, inf, inf, inf]])), \
		#mu = m_par[0], m1 = m_par[1], m2 = m_par[2], m1 = m_par[3], m1 = m_par[4], m1 = m_par[5]
		"4NN_triangular" : (np.array([[0, inf, 0, inf, inf, inf, 0, 0], \
						[inf, inf, var_1NN_mu, inf, m_par[0]/neigbours, inf, var_1NN_mu, var_2NN_mu], \
						[inf, inf, inf, 0, inf, var_1NN_0, var_2NN_0, var_1NN_0], \
						[0, var_1NN_mu, var_2NN_0, inf, inf, var_2NN_0, var_3NN, var_3NN], \
						[0, var_2NN_mu, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN, var_4NN], \
						[0, var_1NN_mu, var_3NN, var_2NN_0, inf, inf, var_2NN_0, var_3NN], \
						[inf, inf, var_2NN_0, var_1NN_0, inf, 0, inf, var_1NN_0], \
						[inf, m_par[0]/neigbours, inf, inf, inf, inf, inf, inf]]), \
					np.array([[0, inf, 0, 0, inf, inf, inf, 0], \
						[inf, inf, var_2NN_0, inf, inf, m_par[0]/neigbours, var_1NN_0, var_1NN_0], \
						[inf, m_par[0]/neigbours, inf, inf, inf, inf, inf, inf], \
						[inf, inf, var_1NN_0, inf, 0, inf, var_1NN_0, var_2NN_0], \
						[0, var_1NN_mu, var_3NN, var_2NN_0, inf, var_1NN_0, var_2NN_0, var_3NN], \
						[0, var_2NN_mu, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN], \
						[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
						[inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0, inf]]), \
					np.array([[0, inf, 0, 0, 0, inf, inf, inf], \
						[inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, m_par[0]/neigbours, inf], \
						[inf, inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0], \
						[inf, m_par[0]/neigbours, inf, inf, inf, inf, inf, inf], \
						[inf, inf, var_2NN_0, var_1NN_0, var_1NN_0, 0, inf, var_1NN_0], \
						[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
						[0, var_2NN_mu, var_3NN, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0], \
						[0, var_1NN_mu, var_2NN_0, var_3NN, var_3NN, var_2NN_0, inf, inf]])), \
		"HT1" : (np.array([[0, m_par[0], m_par[0], 2.0*m_par[0]+m_par[4]], \
						[m_par[0], 2.0*m_par[0], 2.0*m_par[0], 3.0*m_par[0]+m_par[4]], \
						[m_par[0], 2.0*m_par[0]+m_par[1], 2.0*m_par[0], 3.0*m_par[0]+m_par[4]+m_par[1]], \
						[2.0*m_par[0]+m_par[4], 3.0*m_par[0]+m_par[4]+m_par[1], 3.0*m_par[0]+m_par[4], 4.0*m_par[0]+2.0*m_par[4]+m_par[1]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0*m_par[0]+m_par[4]], \
						[m_par[0], 2.0*m_par[0]+m_par[2], 2.0*m_par[0], 3.0*m_par[0]+m_par[4]+m_par[2]], \
						[m_par[0], 2.0*m_par[0]+m_par[4], 2.0*m_par[0]+m_par[2], 3.0*m_par[0]+2.0*m_par[4]+m_par[2]], \
						[2.0*m_par[0]+m_par[4], 3.0*m_par[0]+2.0*m_par[4]+m_par[2], 3.0*m_par[0]+m_par[4]+m_par[2], 4.0*m_par[0]+3.0*m_par[4]+2.0*m_par[2]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0*m_par[0]+m_par[4]], \
						[m_par[0], 2.0*m_par[0]+m_par[2], 2.0*m_par[0]+m_par[1], 3.0*m_par[0]+m_par[4]+m_par[1]+m_par[2]], \
						[m_par[0], 2.0*m_par[0]+m_par[1], 2.0*m_par[0]+m_par[2], 3.0*m_par[0]+m_par[4]+m_par[1]+m_par[2]], \
						[2.0*m_par[0]+m_par[4], 3.0*m_par[0]+m_par[4]+m_par[1]+m_par[2], 3.0*m_par[0]+m_par[4]+m_par[1]+m_par[2], 4.0*m_par[0]+2.0*m_par[4]+2.0*m_par[1]+2.0*m_par[2]]]), \
					np.array([[0, m_par[0], m_par[0], 2.0*m_par[0]+m_par[4]], \
						[m_par[0], 2.0*m_par[0]+m_par[2], 2.0*m_par[0]+m_par[4], 3.0*m_par[0]+2.0*m_par[4]+m_par[2]], \
						[m_par[0], 2.0*m_par[0], 2.0*m_par[0]+m_par[2], 3.0*m_par[0]+m_par[4]+m_par[2]], \
						[2.0*m_par[0]+m_par[4], 3.0*m_par[0]+m_par[4]+m_par[2], 3.0*m_par[0]+2.0*m_par[4]+m_par[2], 4.0*m_par[0]+3.0*m_par[4]+2.0*m_par[2]]])), \
		"HT2" : (np.array([[0, 2.0*m_par[0]/4.0, m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0+m_par[5], m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[5], 2.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0, m_par[0]/4.0, m_par[0]/4.0], \
						[2.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[3], inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/4.0, 3.0*m_par[0]/4.0, 3.0*m_par[0]/4.0], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[5], 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0], \
						[2.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[4], inf, inf, inf, 3.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[4]+m_par[5], 4.0*m_par[0]/4.0+2.0*m_par[4], inf, 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[4]], \
						[2.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[5]+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[5]], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[5], 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[5], 2.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0], \
						[2.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[5], inf, inf, inf, 3.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+2.0*m_par[5], 4.0*m_par[0]/4.0+m_par[5]+m_par[4], inf, 3.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[5]], \
						[2.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[4], 4.0*m_par[0]/4.0+m_par[4]+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[4]], \
						[2.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[1], 3.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[3], 3.0*m_par[0]/4.0, 3.0*m_par[0]/4.0], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0, inf, inf, inf, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[4], inf, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[4], 3.0*m_par[0]/4.0+m_par[5], inf, inf, inf, inf, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0]]), \
					np.array([[0, 2.0*m_par[0]/4.0, m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0+m_par[5], inf, inf, inf, inf, m_par[0]/4.0, m_par[0]/4.0], \
						[inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[4], inf, inf, inf], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[3], inf, inf, inf, inf, inf, inf, inf, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[4], inf, inf], \
						[2.0*m_par[0]/4.0+m_par[5], 4.0*m_par[0]/4.0+m_par[5]+m_par[4]+m_par[3], inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[5], inf], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[2], 2.0*m_par[0]/4.0+m_par[3], 3.0*m_par[0]/4.0+m_par[4]+m_par[3], 3.0*m_par[0]/4.0+m_par[3]+m_par[5], inf, inf, inf, inf, 2.0*m_par[0]/4.0, 2.0*m_par[0]/4.0], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[5], inf, inf], \
						[2.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[4], 3.0*m_par[0]/4.0+m_par[4]+m_par[5], 4.0*m_par[0]/4.0+2.0*m_par[4]+m_par[5], 4.0*m_par[0]/4.0+m_par[4]+2.0*m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[4]+m_par[3], inf], \
						[2.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[4], 4.0*m_par[0]/4.0+m_par[4]+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[2], 3.0*m_par[0]/4.0+m_par[3]], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0, inf, inf], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, 2.0*m_par[0]/4.0+m_par[3], inf]]), \
					np.array([[0, 2.0*m_par[0]/4.0, m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0+m_par[5], inf, inf, inf, 2.0*m_par[0]/4.0, m_par[0]/4.0, inf], \
						[2.0*m_par[0]/4.0, inf, 3.0*m_par[0]/4.0+m_par[3], 4.0*m_par[0]/4.0+m_par[3]+m_par[4], inf, inf, inf, inf, 4.0*m_par[0]/4.0, 3.0*m_par[0]/4.0, inf], \
						[inf, inf, inf, inf, inf, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[4], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/4.0+2.0*m_par[4], inf, inf, inf], \
						[inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[5]+m_par[2], inf, inf, inf, inf], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[3], 2.0*m_par[0]/4.0+m_par[2], 3.0*m_par[0]/4.0+m_par[2]+m_par[4], 3.0*m_par[0]/4.0+m_par[2]+m_par[5], inf, inf, inf, 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[1], inf], \
						[inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[2]+2.0*m_par[5], inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[5]], \
						[2.0*m_par[0]/4.0+m_par[4], 4.0*m_par[0]/4.0+m_par[3]+m_par[4], 3.0*m_par[0]/4.0+m_par[3]+m_par[4], 4.0*m_par[0]/4.0+2.0*m_par[4]+2.0*m_par[1]+2.0*m_par[2], inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[4]+m_par[3], 3.0*m_par[0]/4.0+m_par[1]+m_par[2], inf], \
						[2.0*m_par[0]/4.0, 4.0*m_par[0]/4.0, 3.0*m_par[0]/4.0, 4.0*m_par[0]/4.0+m_par[4]+m_par[3], inf, inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[3], inf], \
						[inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[5], inf, inf, inf, inf, inf, 2.0*m_par[0]/4.0], \
						[m_par[0]/4.0, 3.0*m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[1], 3.0*m_par[0]/4.0+m_par[1]+m_par[2], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[3], 2.0*m_par[0]/4.0+m_par[2], inf]]), \
					np.array([[0, inf, m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[4], inf, m_par[0]/4.0, 2.0*m_par[0]/4.0+m_par[5], inf, 2.0*m_par[0]/4.0, m_par[0]/4.0, inf], \
						[2.0*m_par[0]/4.0, inf, 3.0*m_par[0]/4.0+m_par[2], 4.0*m_par[0]/4.0+2.0*m_par[4]+m_par[2], inf, 3.0*m_par[0]/4.0+m_par[3], 4.0*m_par[0]/4.0+m_par[4]+m_par[5]+m_par[3], inf, 4.0*m_par[0]/4.0+m_par[5], 3.0*m_par[0]/4.0+m_par[4], inf], \
						[inf, 3.0*m_par[0]/4.0, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, 4.0*m_par[0]/4.0+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, 4.0*m_par[0]/4.0+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0]/4.0, inf, 2.0*m_par[0]/4.0+m_par[3], 3.0*m_par[0]/4.0+m_par[4]+m_par[3]+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/4.0+m_par[4], 2.0*m_par[0]/4.0+m_par[5], inf], \
						[2.0*m_par[0]/4.0, inf, 3.0*m_par[0]/4.0+m_par[3]+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[4]+m_par[3]+m_par[5], inf, inf], \
						[2.0*m_par[0]/4.0+m_par[4], inf, 3.0*m_par[0]/4.0+m_par[4]+m_par[3], 4.0*m_par[0]/4.0+2.0*m_par[3]+2.0*m_par[4], inf, inf, inf, inf, 4.0*m_par[0]/4.0+2.0*m_par[4]+m_par[2], 3.0*m_par[0]/4.0+m_par[4]+m_par[5]+m_par[3], inf], \
						[inf, inf, inf, inf, 4.0*m_par[0]/4.0+m_par[5], inf, inf, 4.0*m_par[0]/4.0, inf, inf, 3.0*m_par[0]/4.0], \
						[m_par[0]/4.0, inf, 2.0*m_par[0]/4.0, inf, inf, 2.0*m_par[0]/4.0, inf, inf, 3.0*m_par[0]/4.0+m_par[3], inf, inf], \
						[m_par[0]/4.0, inf, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[3]+m_par[4], inf, 2.0*m_par[0]/4.0, 3.0*m_par[0]/4.0+m_par[3]+m_par[5], inf, 3.0*m_par[0]/4.0+m_par[2], 2.0*m_par[0]/4.0+m_par[3], inf]])) , \
		"HT3" : (np.array([[0, m_par[0]/9, m_par[0]/9, m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, m_par[0]/9, 2.0*m_par[0]/9+m_par[5], 2.0*m_par[0]/9+m_par[5], 2.0*m_par[0]/9+m_par[4], 2.0*m_par[0]/9, inf, inf, inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[4], 2.0*m_par[0]/9+m_par[5], 2.0*m_par[0]/9+m_par[5], 2.0*m_par[0]/9, inf, 2.0*m_par[0]/9], \
						[m_par[0]/9, inf, inf, 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[4], 3.0*m_par[0]/9+m_par[5], inf, inf, 3.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[3], inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, 4.0*m_par[0]/9+m_par[4]], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[2], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3]+m_par[4], 3.0*m_par[0]/9+m_par[3]+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], 3.0*m_par[0]/9, inf, 3.0*m_par[0]/9], \
						[2.0*m_par[0]/9+m_par[5], inf, inf, 3.0*m_par[0]/9+m_par[5]+m_par[3], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[5], inf], \
						[2.0*m_par[0]/9+m_par[4], 3.0*m_par[0]/9+m_par[4]+m_par[3], inf, inf, 4.0*m_par[0]/9+m_par[4]+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[4]+m_par[5]+m_par[3], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[5]+2.0*m_par[3], inf, inf, inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, 4.0*m_par[0]/9+m_par[5]], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf, 3.0*m_par[0]/9], \
						[2.0*m_par[0]/9+m_par[5], inf, inf, 3.0*m_par[0]/9+m_par[3]+m_par[5], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf], \
						[2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, 4.0*m_par[0]/9+2.0*m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[2], 2.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3]+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], 3.0*m_par[0]/9+m_par[4]+m_par[3], 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf, 3.0*m_par[0]/9], \
						[2.0*m_par[0]/9+m_par[4], inf, inf, 3.0*m_par[0]/9+m_par[3]+m_par[4], 4.0*m_par[0]/9+m_par[4]+m_par[5], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[3], 3.0*m_par[0]/9+m_par[4]+m_par[5], inf, inf, 4.0*m_par[0]/9+2.0*m_par[4]+2.0*m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf], \
						[2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, 4.0*m_par[0]/9+2.0*m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[2], 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[1], 4.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+m_par[5]+m_par[4], 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[5]+m_par[4], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+2.0*m_par[5], 4.0*m_par[0]/9+m_par[3], inf, 4.0*m_par[0]/9+m_par[3]], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[2], 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf]]), \
					np.array([[0, m_par[0]/9, m_par[0]/9, m_par[0]/9, 2.0*m_par[0]/9, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[4], 2.0*m_par[0]/9+m_par[5], 2.0*m_par[0]/9+m_par[5], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[3], inf, inf, 3.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[2], inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3]+m_par[4], 3.0*m_par[0]/9+m_par[3]+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5]], \
						[2.0*m_par[0]/9+m_par[4], 3.0*m_par[0]/9+m_par[4], inf, inf, 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+2.0*m_par[4]+2.0*m_par[5]+2.0*m_par[3], inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4]], \
						[2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5]], \
						[2.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[2], 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf]]), \
					np.array([[0, m_par[0]/9, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[4], inf, inf, 2.0*m_par[0]/9, 2.0*m_par[0]/9, inf], \
						[inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf], \
						[inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[2], inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[1], 3.0*m_par[0]/9+m_par[1]+m_par[2]+m_par[4], inf, inf, 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3], inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+m_par[5], inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[4], inf], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[1], inf, inf, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[2], 3.0*m_par[0]/9+m_par[1]+m_par[2]+m_par[4], inf, inf, 3.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[5], inf], \
						[inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+2.0*m_par[5]+m_par[2], inf, inf, inf, inf], \
						[inf, inf, inf, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[2]+2.0*m_par[5], inf, inf, inf], \
						[2.0*m_par[0]/9+m_par[4], 3.0*m_par[0]/9+m_par[1]+m_par[2], inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[1]+m_par[2]+m_par[4], 4.0*m_par[0]/9+2.0*m_par[1]+2.0*m_par[2]+2.0*m_par[4], inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf], \
						[inf, inf, inf, inf, inf, 4.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[2]+2.0*m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+2.0*m_par[2]+2.0*m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3], inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf]]), \
					np.array([[0, m_par[0]/9, inf, inf, inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[5], inf, inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[5], inf, inf, m_par[0]/9, 2.0*m_par[0]/9+m_par[4], inf, inf, 2.0*m_par[0]/9, 2.0*m_par[0]/9, inf], \
						[m_par[0]/9, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], inf], \
						[m_par[0]/9, inf, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, 2.0*m_par[0]/9, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], inf], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[3], inf, inf, inf, inf, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, inf, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, 2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[3]+m_par[4], inf, inf, 3.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[2], inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, 3.0*m_par[0]/9+m_par[2], 4.0*m_par[0]/9+2.0*m_par[4]+m_par[2], inf, inf, 4.0*m_par[0]/9+m_par[5], 4.0*m_par[0]/9+m_par[5], inf], \
						[2.0*m_par[0]/9, 3.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3], 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[4], 4.0*m_par[0]/9+m_par[5], inf], \
						[m_par[0]/9, 2.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 2.0*m_par[0]/9+m_par[3], 3.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, 3.0*m_par[0]/9+m_par[5], 3.0*m_par[0]/9+m_par[4], inf], \
						[2.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf], \
						[2.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[3]+m_par[5], inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf], \
						[2.0*m_par[0]/9+m_par[4], 3.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9+m_par[4]+m_par[3], 4.0*m_par[0]/9+2.0*m_par[3]+2.0*m_par[4]+m_par[5], inf, inf, 4.0*m_par[0]/9+m_par[3]+m_par[4]+m_par[5], 4.0*m_par[0]/9+m_par[2]+2.0*m_par[4], inf], \
						[inf, inf, inf, inf, 4.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 3.0*m_par[0]/9, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf], \
						[inf, inf, 3.0*m_par[0]/9, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, inf], \
						[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, 4.0*m_par[0]/9], \
						[inf, inf, inf, 3.0*m_par[0]/9, inf, inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[4], inf, inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf, 4.0*m_par[0]/9+m_par[5], inf, inf, inf]]))
	}

	matrixes = list(matrix_dict.get(model))
	assert (matrixes is not None), "Error! This model is not in the database"
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
