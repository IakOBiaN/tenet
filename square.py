from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import TensorNetworks as tn


inf = -1e8
constant = 1.
interactions = [1.]
temperature = 1.
l_vector = []
chi_number = 24
chi_min = 1e-8
method_tolerance = 1e-8

def build_matrix (model, temp, m_par, neigbours = 8.0):

	if len(m_par) < 10:
		m_par = m_par + [0.0]*(10-len(m_par))

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrix_dict = {
		"langmuir" : (np.array([[0.0, m_par[0]/neigbours],[m_par[0]/neigbours, -m_par[1]+m_par[0]/(neigbours/2.0)]]) ,) * 3,
		"ising" : (np.array([[(m_par[1]-m_par[0]/(neigbours/2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1]+m_par[0]/(neigbours/2.0))]]), ) * 3,
		"hard-disk" : (np.array([[0.0, m_par[0]/(neigbours/2.0)],[m_par[0]/(neigbours/2.0), -1000000.0+m_par[0]]]), ) * 3,
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

def simulate(method = "trg", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0]*10):

	tensors = build_matrix(model, temp, m_par)
	tensors = tn.build_tensor(tensors, lattice)
	#tensors = tn.build_triangles_tensor(model, temp, m_par)

	scale = 0.0
	old_scale = -1.0
	if lattice == "triangle":
		nodes = 1.0
	else:
		nodes = 2.0

	i = 0
	for i in range(300):
		if method == "trg":
			(tensors, scale) = tn.trg_step(tensors, scale, chi_number, chi_min, lattice)
		elif method == "hotrg":
			(tensors, scale) = tn.hotrg_step(tensors, scale, chi_number, chi_min, lattice)
		else:
			assert False, "Error! There is no such method."
		if abs(old_scale - scale/4.0) < method_tolerance:
			break
		else:
			old_scale = scale
	if i > 250:
		print("Warning! More than 250 iterations")
	nodes *= 4.0**(i+1)
	norm = np.einsum("abab->",tensors[0])
	if norm < 0:
		norm = -norm
	return (scale+log(norm))/(nodes/(constant*temp))

def coverage_old(method, model, lattice, temp = 1., m_par = [0.0]*10):
	result = derivative(lambda x: simulate(method, model, lattice, temp, [x]+m_par[1:]), m_par[0], n=1, dx=1e-3)
	return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
	return result

def heat_capasity(method, model, lattice, temp = 1., m_par = [0.0]*10):
	result = derivative(lambda x: simulate(method, model, lattice, x, m_par), temp, n=2, dx=1e-3)
	return result

def enthropy_old(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,x,field,int), temp, n=1, dx=1e-5)
	return result

def coverage(method, model, lattice, temp = 1., m_par = [0.0]*10):
	BTP = []
	mu_step = 1e-3
	for mu_TRG in [m_par[0] - mu_step, m_par[0] + mu_step]:
		lnZ = simulate(method, model, lattice, temp, [mu_TRG] + m_par[1:])
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(mu_step*2.0)
	return result

def entropy(method, model, lattice, temp = 1., m_par = [0.0]*10):
	BTP = []
	temp_step = 1e-3
	for temp_TRG in [temp - temp_step, temp + temp_step]:
		lnZ = simulate(method, model, lattice, temp_TRG, m_par)
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(temp_step*2.0)
	return result

def full(method, model, lattice, temp = 1., m_par = [0.0]*10):
	BTP_mu = []
	BTP_temp = []
	step = 1e-3
	for mu_TRG in [m_par[0] - step, m_par[0] + step]:
		lnZ = simulate(method, model, lattice, temp, [mu_TRG] + m_par[1:])
		BTP_mu.append(lnZ)
	for temp_TRG in [temp - step, temp, temp + step]:
		lnZ = simulate(method, model, lattice, temp_TRG, m_par)
		BTP_temp.append(lnZ)

	cov = -(BTP_mu[0]-BTP_mu[1])/(step*2.0)
	ent = -(BTP_temp[0]-BTP_temp[2])/(step*2.0)
	sus = (BTP_mu[0]-2.0*BTP_temp[1]+BTP_mu[1])/(step**2.0)
	cap = (BTP_temp[0]-2.0*BTP_temp[1]+BTP_temp[2])/(step**2.0)
	return cov, ent, sus, cap

method = "trg"
model = "ising"
lattice = "tr_to_sqr"
temp_square = 2.0/log(1+sqrt(2))
#temp_hex = 4.0/log(3)
#temp = temp_hex
#temp = 4.0/log(3) + 1e-7
temp = 1.0
mu = 1.0
chi_number = 24
#for temp in np.arange(2.0, 4.41, 0.05):
for mu in np.arange(8.0, -8.01, -0.5):
	m_par = [mu/6.0, 1.0, 0.0, 0.0, 0.0, 0.0]
	#m_par = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
	#coef = 1.0/8.00
	#print(-mu, coef*coverage(method, model, lattice, temp, m_par))
	print(mu, simulate(method, model, lattice, temp, m_par))
	#result = full(method, model, lattice, temp, m_par)
	#print(-mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3])
	#print(-mu, coef*result[2])
