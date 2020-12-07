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
	tensors = tn.build_tensor(tensors, lattice)
	scale = 0.0
	old_scale = -1.0
	if lattice == "triangle":
		nodes = 1.0
	else:
		nodes = 2.0

	for i in range(2):
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


	#tensors[0] = np.einsum("abcd, cfgh, ihkl, mdip -> mabfkgpl", tensors[0],tensors[0],tensors[0],tensors[0]).reshape(4,4,4,4)
	#tensors[0] = np.einsum("abcd, cfgh, ihkl, mdip -> mabfkgpl", tensors[0],tensors[0],tensors[0],tensors[0]).reshape(16,16,16,16)
	#norm = abs(np.trace(np.trace(tensors[0], axis1 = 0, axis2 = 2)))
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

"""temperature = 1.
muu = 0.00001
for temperature in np.arange(2.2,2.3,0.001):
	interactions = [1.0,0.0,0.0]
	method = TRG
	size = 16
	#print(muu,coverage(method,'hard_triangle2',size, temperature, muu/10.0, interactions))#,heat_capasity(method,'BC_ising',size, T, muu, interactions))
	print(temperature,heat_capasity(method,'ising',size, temperature, muu, interactions))
	#print(muu,coverage(method,'hard_triangle',size, temperature, muu/3.0, interactions))"""

"""temperature = 2.0/log(1+sqrt(2))
muu = 0.0
interactions = [1.0,0.0,0.0]
method = TRG
size = 16
for size in np.arange(10, 66, 5):
#for temperature in np.arange(0.5,3,0.1):
	print(size,method('ising',size, temperature, muu, interactions))"""

method = "trg"
model = "ising"
lattice = "hexagonal"
temp_square = 2.0/log(1+sqrt(2))
temp_hex = 4.0/log(3)
temp = temp_hex
mu = 0.0
m_par = [mu, 1.0]
for size in np.arange(2, 36, 1):
	chi_number = size
	print(size, simulate(method, model, lattice, temp, m_par))


"""method = "trg"
model = "langmuir"
lattice = "hexagonal"
temp = 2
mu = 0.0
chi_number = 16
for mu in np.arange(10.0,40.0,0.5):
	m_par = [mu, 0.0]
	print(mu,coverage(method, model, lattice, temp, m_par))"""
