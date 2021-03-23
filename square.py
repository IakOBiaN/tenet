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

def build_matrix (model, temp, m_par, neigbours = 4.0):

	if len(m_par) < 10:
		m_par = m_par + [0.0]*(10-len(m_par))

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrix_dict = {
		"langmuir" : (np.array([[0.0, m_par[0]/neigbours],[m_par[0]/neigbours, -m_par[1]+m_par[0]/(neigbours/2.0)]]) ,) * 2,
		"ising" : (np.array([[(m_par[1]-m_par[0]/(neigbours/2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1]+m_par[0]/(neigbours/2.0))]]), ) * 2,
		"hard-disk" : (np.array([[0.0, m_par[0]/(neigbours/2.0)],[m_par[0]/(neigbours/2.0), -1000000.0+m_par[0]]]), ) * 2,
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
						[2.0*m_par[0]+m_par[4], 3.0*m_par[0]+m_par[4]+m_par[2], 3.0*m_par[0]+2.0*m_par[4]+m_par[2], 4.0*m_par[0]+3.0*m_par[4]+2.0*m_par[2]]]))
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

	for i in range(100):
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
	if i > 50:
		print("Warning! More than 50 iterations")
	nodes *= 4.0**(i+1)

	return scale/(nodes/(constant*temp))

def coverage(method, model, lattice, temp = 1., m_par = [0.0]*10):
	result = derivative(lambda x: simulate(method, model, lattice, temp, [x]+m_par[1:]), m_par[0], n=1, dx=1e-3)
	return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
	return result

def heat_capasity(method, model, lattice, temp = 1., m_par = [0.0]*10):
	result = derivative(lambda x: simulate(method, model, lattice, x, m_par), temp, n=2, dx=1e-3)
	return result

def enthropy(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,x,field,int), temp, n=1, dx=1e-5)
	return result


method = "trg"
model = "langmuir"
lattice = "complex_to_sqr"
temp_square = 2.0/log(1+sqrt(2))
temp_hex = 4.0/log(3)
temp = temp_hex
temp = 1.0
mu = 1.0
chi_number = 16
for mu in np.arange(-20.0,50.01,1.0):
	m_par = [mu, 5.0, inf, inf, inf, inf]
	print(mu, coverage(method, model, lattice, temp, m_par))
