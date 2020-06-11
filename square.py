from math import exp, log
import numpy as np
from scipy.misc import derivative

def get_simple_tensor_2 (model, beta = 1., field = 0, int = 0):

	tensor_dict = {
		"langmuir" : np.array([[0.0, field/2.0],[field/2.0, -int+field]]),
		"ising" : np.array([[(int-field), (-int)],[(-int), (int+field)]]),
		"hard-disk" : np.array([[0.0, field/2.0],[field/2.0, -1000000.0+field]])
	}

	tensor_2 = tensor_dict.get(model)
	assert (tensor_2 is not None), "Error! This model is not in the database"
	tensor_2 *= beta
	tensor_2 = np.array([np.exp(line) for line in tensor_2])
	return tensor_2

def create_cd(dim,elem):
	cd = np.zeros((elem,) * dim)
	for i in range(elem):
		cd[((i,)*dim)] = 1.
	return cd

interactions = 5.
temperature = 200.

def eigenvalues(size, hor_tensor_2, vert_tensor_2, tolerance = 1e-10, vectors = False):
	elem = hor_tensor_2.shape[0]
	l_vector = np.random.rand(elem**size).reshape([elem for i in range(size)])
	dop_vector = l_vector + 1e-1
	vert_tensor_2 = np.einsum("kl,ijk,lmn->ijmn",vert_tensor_2,create_cd(3,elem),create_cd(3,elem))
	error = 1.0
	eigenvalue = None
	while error > tolerance:
		for i in range(size):
			l_vector = np.tensordot(l_vector,hor_tensor_2, axes=([0],[0]))
		for i in range(size):
			l_vector = np.tensordot(l_vector,vert_tensor_2, axes=([0,-1],[0,2]))
		eigenvalue = np.linalg.norm(l_vector)
		l_vector /= eigenvalue
		error = np.linalg.norm(l_vector - dop_vector)
		l_vector = 0.5*(l_vector) + 0.5*(dop_vector)
		dop_vector = l_vector
	return eigenvalue

def simple_TM(model,size, beta = 1., field = 0, int = 0):
	tensor = get_simple_tensor_2(model, beta,field, int = interactions)
	return log(eigenvalues(size, tensor, tensor))/beta/(2.*size)

def simple_hierarchical(model,size, beta = 1., field = 0, int = 0):
	tensor = get_simple_tensor_2(model, beta,field, int = interactions)

	number_of_steps = 100
	Z = np.empty((number_of_steps+1))
	Z[0] = tensor.max()
	tensor = tensor/Z[0]
	lnZ_list = []
	cd3 = create_cd(3, tensor.shape[0])
	cd4 = create_cd(4, tensor.shape[0])

	cd6 = create_cd(6, tensor.shape[0])

	for i in np.arange(1,(number_of_steps+1),1):
		edges = (1 + size*2)**2
		dop_tensor = create_cd(size+2,tensor.shape[0])
		dop_tensor_2 = create_cd(size+2,tensor.shape[0])

		doubled_tensor = np.einsum("ij,aic->ajc",tensor,cd3)
		doubled_tensor = np.einsum("ij,abi->abj",tensor,doubled_tensor)
		doubled_tensor = np.einsum("ijk,ajk->ai",doubled_tensor,cd3)

		for ii in range(size):
			dop_tensor = np.tensordot(dop_tensor,doubled_tensor, axes=([1],[0]))
		dop_tensor = np.tensordot(dop_tensor,tensor, axes=([1],[0]))

		for ii in range(size+(size-1)):
			dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-1],[0]))
			for j in range(size):
				dop_tensor = np.tensordot(dop_tensor,cd3, axes=([-2-j],[0]))
				dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-2],[0]))
				dop_tensor = np.tensordot(dop_tensor,cd3, axes=([-1,-3],[0,1]))
				dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-2],[0]))

		for ii in range(size):
			dop_tensor = np.tensordot(dop_tensor,doubled_tensor, axes=([1],[0]))
		dop_tensor = np.tensordot(dop_tensor,tensor, axes=([1],[0]))
		tensor = np.tensordot(dop_tensor,dop_tensor_2, axes=(np.arange(1,size+2,1),np.arange(1,size+2,1)))

		Z[i] = tensor.max()
		tensor = tensor/Z[i]

		xxx = np.trace(tensor)
		if xxx == 0:
			break

		lnZ = np.log(xxx)
		lnZ /= (edges**i)
		logZ_powers_sum = sum(sorted([log(Z[j]) / (edges**j) for j in range(0,i+1)]))
		lnZ += logZ_powers_sum

		lnZ_list.append(lnZ)
		if len(lnZ_list) > 3 and abs(lnZ_list[-1] - lnZ_list[-2]) < 1e-9 and abs(lnZ_list[-1] - lnZ_list[-3]) < 1e-9:
			break
	return lnZ_list[-1]/beta

def coverage(method,model,size, beta = 1., field = 0, int = 0):
	result = derivative(lambda x: method(model,size,beta,x,int), field, n=1, dx=0.001)
	return result

def heat_capasity(method,model,size, beta = 1., field = 0, int = 0):
	res = derivative(lambda x: -method(model,size,beta,x,int), T, n=2, dx=1e-5)
	return res

for muu in np.arange(-10,20,0.1):
	method = simple_hierarchical
	print(muu,coverage(method,'langmuir',15, 1./(0.008314*temperature), muu, interactions))
