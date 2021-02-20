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
						[m_par[0]/3.0, inf, inf, inf, inf]]))
	}

	matrixes = list(matrix_dict.get(model))
	assert (matrixes is not None), "Error! This model is not in the database"
	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i]/(constant*temp)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])
	return matrixes

def simulate(method = "trg", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0, 0.0]):

	tensors = build_matrix(model, temp, m_par)
	#lattice = "hexagonal"
	tensors = tn.build_tensor(tensors, lattice)
	#lattice = "square"
	scale = 0.0
	old_scale = -1.0
	if lattice == "triangle":
		nodes = 1.0
	else:
		nodes = 2.0

	for i in range(100):
		if method == "trg":
			(tensors, scale) = tn.trg_step(tensors, scale, chi_number, chi_min, lattice)
		else:
			assert False, "Error! There is no such method."
		if abs(old_scale - scale/9.0) < method_tolerance:
			break
		else:
			old_scale = scale
	if i > 50:
		print("Warning! More than 50 iterations")
	nodes *= 9.0**(i+1)


	"""norm = abs(np.einsum("abc, cba -> ", tensors[0], tensors[1]))
	if norm != 0:
		for i, ten in enumerate(tensors):
			tensors[i] = ten/norm
		scale += np.log(norm)"""

	return scale/(nodes/(constant*temp))

def coverage(method, model, lattice, temp = 1., m_par = [0.0, 0.0]):
	result = derivative(lambda x: simulate(method, model, lattice, temp, [x, m_par[1]]), m_par[0], n=1, dx=1e-3)
	return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
	return result

def heat_capasity(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,x,field,int), temp, n=2, dx=1e-5)
	return result

def enthropy(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,x,field,int), temp, n=1, dx=1e-5)
	return result


method = "trg"
model = "hard_triangles"
lattice = "square"
temp_square = 2.0/log(1+sqrt(2))
temp_hex = 4.0/log(3)
temp = temp_hex
temp = 1.0
mu = 25.0
for size in range(32,34,1):
	print("SIZE=", size)
	for mu in np.arange(-10, 10, 0.5):
		chi_number = 32
		m_par = [mu/3.0, 0.0]
		print(mu, coverage(method, model, lattice, temp, m_par))
