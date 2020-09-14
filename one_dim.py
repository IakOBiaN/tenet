from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg

constant = 1.
interactions = [1.]
temperature = 1.
l_vector = []

def get_simple_tensor_2 (model, temp = 1., field = 0, int = [0, 0, 0]):

	max = 3
	for i in range(max-len(int)):
		int.append(0.0)

	tensor_dict = {
		"langmuir" : np.array([[0.0, field/4.0],[field/4.0, -int[0]+field/2.0]]),
		"ising" : np.array([[(int[0]-field/2.0), (-int[0])],[(-int[0]), (int[0]+field/2.0)]]),
		"hard-disk" : np.array([[0.0, field/2.0],[field/2.0, -1000000.0+field]]),
		"BC_ising" : np.array([[int[1]+field/4.0, -int[1], int[1]+int[2]/4.0 + 3.0*field/8.0, -int[1]-int[2]/4.0+field/8.0,int[1]-int[2]/4.0+field/8.0,-int[1]+int[2]/4.0-field/8.0], \
							   [-int[1],int[1]-field/4.0,-int[1]+int[2]/4.0+field/8.0,int[1]-int[2]/4.0-field/8.0,-int[1]-int[2]/4.0-field/8.0,int[1]+int[2]/4.0-3.0*field/8.0], \
							   [int[1]+int[2]/4.0+3.0*field/8.0,-int[1]+int[2]/4.0+field/8.0,int[0]+int[1]+int[2]/2.0+field/2.0,int[0]-int[1]+field/8.0,-int[0]+int[1]+field/4.0,-int[0]-int[1]+int[2]/2.0], \
							   [-int[1]-int[2]/4.0+field/8.0,int[1]-int[2]/4.0-field/8.0,int[0]-int[1]+field/4.0,int[0]+int[1]-int[2]/2.0,-int[0]-int[1]-int[2]/2.0,-int[0]+int[1]-field/4.0], \
							   [int[1]-int[2]/4.0+field/8.0,-int[1]-int[2]/4.0-field/8.0, -int[0]+int[1]+field/4.0,-int[0]-int[1]-int[2]/2.0,int[0]+int[1]-int[2]/2.0,int[0]-int[1]-field/4.0], \
							   [-int[1]+int[2]/4.0-field/8.0,int[1]+int[2]/4.0-3.0*field/8.0,-int[0]-int[1]+int[2]/2.0,-int[0]+int[1]-field/4.0,int[0]-int[1]-field/4.0,int[0]+int[1]+int[2]/2.0-field/2.0]])
	}

	tensor_2 = tensor_dict.get(model)
	assert (tensor_2 is not None), "Error! This model is not in the database"
	tensor_2 *= 1./(constant*temp)
	tensor_2 = np.array([np.exp(line) for line in tensor_2])
	#tensor_2 = np.longdouble(tensor_2)
	return tensor_2

def create_cd(dim,elem):
	cd = np.zeros((elem,) * dim)
	for i in range(elem):
		cd[((i,)*dim)] = 1.
	return cd

def eigenvalues_degree(size, tensor, tolerance = 1e-10, vectors = False):
	elem = tensor.shape[0]
	global l_vector
	#global r_vector
	if (len(l_vector) == 0):
		#l_vector = np.longdouble(np.random.rand(elem**size).reshape([elem for i in range(size)]))
		l_vector = np.array([1.] * elem)
		#r_vector = np.array([1.] * elem)
	dop_vector = l_vector + 1e-1
	#dop_r_vector = r_vector + 1e-1
	error = 1.0
	eigenvalue = None
	count = 0
	while error > tolerance:
		count += 1
		l_vector = np.einsum("i,ij->j",l_vector,tensor)
		#r_vector = np.einsum("i,ji->j",r_vector,tensor)
		#print(tensor)

		eigenvalue = np.linalg.norm(l_vector)
		l_vector /= eigenvalue

		error = np.linalg.norm(l_vector - dop_vector)
		#l_vector = 0.5*(l_vector) + 0.5*(dop_vector)
		dop_vector = l_vector
		#print(error,count,eigenvalue)
	return eigenvalue

def simple_TM(model,size, temp = 1., field = 0, int = [0]):

	inf = -1e10
	if (model == "AVM"):
		field /= 2.0
		tensor = np.array([[0, field, 2.0*field, inf, inf, inf], \
						[inf, inf, inf, 3.0*field, inf, inf], \
						[inf, inf, inf, inf, 3.0*field, 4.0*field], \
						[2.0*field, 3.0*field, 4.0*field, inf, inf, inf], \
						[field, 2.0*field, 3.0*field, inf, inf, inf], \
						[inf, inf, inf, 4.0*field, inf, inf]])
	if (model == "SSA"):
		field /= 2.0
		tensor = np.array([[0, field, inf, inf], \
						[inf, inf, 2.0*field, inf], \
						[inf, inf, inf, 2.0*field], \
						[field, 2.0*field, inf, inf]])

	if (model == "lang"):
		field /= 2.0
		tensor = np.array([[0, field], \
						[field, 2.0*field]])
	if (model == "lang2"):
		field /= 2.0
		tensor = np.array([[0, field, field, 2.0*field], \
						[field, 2.0*field, 2.0*field, 3.0*field], \
						[field, 2.0*field, 2.0*field, 3.0*field], \
						[2.0*field, 3.0*field, 3.0*field, 4.0*field]])

	tensor *= 1./(constant*temp)
	tensor = np.array([np.exp(line) for line in tensor])

	return log(eigenvalues_degree(size, tensor))/(1./(constant*temp))/2.0

def coverage(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: method(model,size,temp,x,int), field, n=1, dx=1e-3)
	return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: method(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
	return result

def heat_capasity(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: method(model,size,x,field,int), temp, n=2, dx=1e-4)
	return result

def enthropy(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: method(model,size,x,field,int), temp, n=1, dx=1e-5)
	return result

temperature = 1.
for muu in np.arange(-5.0,10,0.1):
	interactions = [0.0,0.0,0.0]
	method = simple_TM
	size = 1
	print(muu,coverage(method,'AVM',size, temperature, muu/3.0, interactions))
