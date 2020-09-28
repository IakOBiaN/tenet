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

def TRG(model,max_count = 64, temp = 1., field = 0, int = [0], max_value = 1e-8, atol_lnZ = 1e-9, contractions_count = 300, lattice = "triangle"):

	"""tensor_0 = get_simple_tensor_2(model, temp,field, int = interactions)
	n = tensor_0.shape[0]
	cd = create_cd(3, n) #SVD of cd is cd
	tensor = np.einsum("ab,ibk->iak",tensor_0,cd)
	if (lattice == "square"):
		tensor = np.einsum("abc,bef,ejk,jcn->afkn",tensor,tensor,tensor,tensor)
	elif (lattice == "triangle"):
		tensor = np.einsum("abc,dfb,icf->adi",tensor,tensor,tensor)
		tensor = np.einsum("abi,ijk->abjk", tensor,cd)"""
	tensor = 0
	if (model == "hard_triangle"):
		inf = -1e8
		field /= 6.0
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
	if (model == "hard_triangle2"):
		inf = -1e8
		field /= 6.0
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
	if (model == "hard_triangle3"):
		inf = -1e8
		field /= 6.0
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
	two2 = np.array([np.exp(line) for line in two])
	three *= 1./(constant*temp)
	three = np.array([np.exp(line) for line in three])
	cd = create_cd(3, one.shape[0])
	one = np.einsum("ab,iak->ibk",one,cd)
	two = np.einsum("ab,iak->ibk",three,cd)
	three = np.einsum("ab,ibk->iak",two2,cd)
	tensor = np.einsum("abc,deb,ice->adi",one,two,three)
	tensor = np.einsum("abi,ijk->abjk", tensor,cd)


	Z = np.empty((contractions_count+1))
	Z[0] = sqrt(tensor.max())
	tensor = tensor/Z[0]
	lnZ_list = []
	temp_count = max_count

	for i in np.arange(1,(contractions_count+1),1):

		U_1,S_1,V_1 = split_by_svd(tensor,[0, 1],[2, 3],reps=max_value, r_max = max_count)
		U_2,S_2,V_2 = split_by_svd(tensor,[0, 3],[1, 2],reps=max_value, r_max = max_count)

		S_1 = np.sqrt(S_1)
		U_1 = np.einsum("abi,i->abi", U_1, S_1)
		V_1 = np.einsum("ibc,i->ibc",V_1, S_1)
		S_2 = np.sqrt(S_2)
		U_2 = np.einsum("abi,i->abi", U_2, S_2)
		V_2 = np.einsum("ibc,i->ibc",V_2, S_2)
		dop_tensor_1 = np.tensordot(V_2,V_1, ([1], [2]))
		dop_tensor_2 = np.tensordot(U_1,U_2, ([1],[1]))
		tensor = np.tensordot(dop_tensor_1, dop_tensor_2, ([1,3],[0,2]))
		tensor = np.swapaxes(tensor,2,3)
		Z[i] = sqrt(tensor.max())
		tensor = tensor/Z[i]
		xxx = abs(np.trace(np.trace(tensor,axis1=0, axis2=2)))
		if xxx == 0.0:
			print("Warning! Tensor trace is zero!")
			break;
		lnZ = log(abs(xxx))
		lnZ /= (2.0**i)
		logZ_powers_sum = sum(sorted([log(Z[j]) / (2.0**j) for j in range(0,i+1)]))
		lnZ += logZ_powers_sum

		lnZ_list.append(lnZ)
		if len(lnZ_list) > 3 and abs(lnZ_list[-1] - lnZ_list[-2]) < atol_lnZ and abs(lnZ_list[-1] - lnZ_list[-3]) < atol_lnZ:
			break

	return lnZ_list[-1]/(1./(constant*temp)*1.0)
	if (lattice == "square"):
		return lnZ_list[-1]/(1./(constant*temp)*2.0)
	elif (lattice == "triangle"):
		return lnZ_list[-1]/(1./(constant*temp)*3.0/2.0)

def matricise(t, legs1, legs2):
	n1 = np.prod([t.shape[i] for i in legs1])
	n2 = np.prod([t.shape[i] for i in legs2])
	return np.moveaxis(t, list(legs1) + list(legs2), list(range(len(t.shape)))).reshape(n1, n2)

def split_by_svd(t, legs1, legs2, reps=0.0, r_max=None):
	if legs1 is not None:
		legs1 = [i % len(t.shape) for i in legs1]
	if legs2 is not None:
		legs2 = [i % len(t.shape) for i in legs2]

	if legs2 is None:
		legs2 = sorted(list(set(range(len(t.shape))) - set(legs1)))
	elif legs1 is None:
		legs1 = sorted(list(set(range(len(t.shape))) - set(legs2)))

	assert len(legs1) + len(legs2) == len(t.shape), (t.shape, legs1, legs2)

	shape1 = [t.shape[i] for i in legs1]
	shape2 = [t.shape[i] for i in legs2]
	m = matricise(t, legs1, legs2)

	if r_max is not None and r_max <= min(m.shape)-1:
		r_max = min(min(m.shape)-1,r_max)
		U, S, V = scipy.sparse.linalg.svds(m,k=r_max)
	else:
		U, S, V = scipy.linalg.svd(m,full_matrices = False, check_finite=False)

	r = len([1 for s in S if s / S.max() > reps])
	if r < 1:
		r = 1
	if r_max is not None and r > r_max:
		r = r_max

	sort = np.argsort(S)[-r:]

	U = U[:, sort]
	S = S[sort]
	V = V[sort, :]

	return U.reshape(shape1 + [r]), S, V.reshape([r] + shape2)

def eigenvalues_degree(size, hor_tensor_2, vert_tensor_2, tolerance = 1e-10, vectors = False):
	elem = hor_tensor_2.shape[0]
	global l_vector
	if (len(l_vector) == 0):
		#l_vector = np.longdouble(np.random.rand(elem**size).reshape([elem for i in range(size)]))
		l_vector = np.array([1.] * (elem**size)).reshape([elem for i in range(size)])
	dop_vector = l_vector + 1e-1
	vert_tensor_2 = np.einsum("kl,ijk,lmn->ijmn",vert_tensor_2,create_cd(3,elem),create_cd(3,elem))
	error = 1.0
	eigenvalue = None
	count = 0
	while error > tolerance:
		count += 1
		for i in range(size):
			l_vector = np.tensordot(l_vector,hor_tensor_2, axes=([0],[0]))
		for i in range(size):
			l_vector = np.tensordot(l_vector,vert_tensor_2, axes=([0,-1],[2,0]))
		eigenvalue = np.linalg.norm(l_vector)
		l_vector /= eigenvalue
		error = np.linalg.norm(l_vector - dop_vector)
		l_vector = 0.5*(l_vector) + 0.5*(dop_vector)
		dop_vector = l_vector
		print(error,count,eigenvalue)
	return eigenvalue

def simple_TM(model,size, temp = 1., field = 0, int = [0]):
	tensor = get_simple_tensor_2(model, temp,field, int = interactions)
	return log(eigenvalues_degree(size, tensor, tensor))/(1./(constant*temp))/(1.*size)

def simple_hierarchical(model,size, temp = 1., field = 0, int = [0]):
	tensor = get_simple_tensor_2(model, temp,field, int = interactions)

	number_of_steps = 100
	Z = np.empty((number_of_steps+1))
	Z[0] = tensor.max()
	tensor = tensor/Z[0]
	lnZ_list = []
	cd3 = create_cd(3, tensor.shape[0])
	cd4 = create_cd(4, tensor.shape[0])

	#cd6 = create_cd(6, tensor.shape[0])
	tensor = np.einsum("ab,ibk->iak",tensor,cd3)
	#triangle
	#tensor = np.einsum("abc,dfb,icf->adi",tensor,tensor,tensor)
	#square
	tensor = np.einsum("abc,bef,ejk,jcn->afkn",tensor,tensor,tensor,tensor)

	for i in np.arange(1,(number_of_steps+1),1):
		"""edges = (1 + size*2)**2
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
			#print("xxx",i)
			break
		"""


		#serpinski classic
		if False:
			edges = 3
			tensor = np.einsum("abc,aef,ibf->eic",tensor,tensor,tensor)

		#serpinski with triangle
		if False:
			edges = 4
			tensor = np.einsum("abc,bef,cfk,fbc->aek",tensor,tensor,tensor,tensor)

		#serpinski classic double
		if False:
			edges = 10
			tensor_one = np.einsum("abc,cef->abef",tensor,tensor)
			tensor_one = np.einsum("abcd,dfg->abcfg",tensor_one,tensor)
			tensor_one = np.einsum("abcdi,ijk->abcdjk",tensor_one,tensor)

			tensor_one = np.einsum("abidef,ijk->abjkdef",tensor_one, cd3)
			tensor_one = np.einsum("aikdefg,ijk->ajdefg",tensor_one,tensor)
			tensor_one = np.einsum("abcief,ijk->abcjkef",tensor_one, cd3)
			tensor_one = np.einsum("abikefg,ijk->abjefg",tensor_one,tensor)
			tensor_one = np.einsum("abcikf,ijk->abcjf",tensor_one,tensor)

			tensor_one = np.einsum("abide,ijk->abjkde",tensor_one, cd3)
			tensor_one = np.einsum("aikdef,ijk->ajdef",tensor_one,tensor)
			tensor_one = np.einsum("abike,ijk->abje",tensor_one,tensor)
			tensor = np.einsum("aikd,ijk->ajd",tensor_one,tensor)

		#square 3 by 3 full NBC
		if False:
			edges = 9
			tensor_one = np.einsum("abid,ijkl->abjkld",tensor,cd4)
			tensor_one = np.einsum("abcdji,ijkl->abcdkl",tensor_one,tensor)
			tensor_one = np.einsum("abcdif,ijkl->abcdjklf",tensor_one,cd4)
			tensor_one = np.einsum("abcdefij,ijkl->abcdefkl",tensor_one,tensor)

			tensor_one = np.einsum("aildefgh,ijkl->ajkdefgh",tensor_one,tensor)
			tensor_one = np.einsum("abcilfgh,ijkl->abcjkfgh",tensor_one,tensor)
			tensor_one = np.einsum("abcdeilh,ijkl->abcdejkh",tensor_one,tensor)

			tensor_one = np.einsum("abijefgh,ijkl->abklefgh",tensor_one,cd4)
			tensor_one = np.einsum("abcdijgh,ijkl->abcdijgh",tensor_one,cd4)

			tensor_one = np.einsum("aildefgh,ijkl->ajkdefgh",tensor_one,tensor)
			tensor_one = np.einsum("abjilfgh,ijkl->abkfgh",tensor_one,tensor)
			tensor = np.einsum("abjilf,ijkl->abkf",tensor_one,tensor)

		#square 3 by 3 PBC
		if False:
			edges = 7
			tensor_one = np.einsum("alcd,ijkl->aijkcd",tensor,tensor)
			tensor_one = np.einsum("abcief,ijkl->abcjklef",tensor_one,tensor)
			tensor_one = np.einsum("abcdejih,ijkl->abcdeklh",tensor_one,tensor)
			tensor_one = np.einsum("abcdefjh,ijkl->abcdefiklh",tensor_one,tensor)
			tensor_one = np.einsum("abcdifkhoj,ijkl->abcdlfho",tensor_one,tensor)
			tensor_one = np.einsum("abcdjigh,ijkl->abcdklgh",tensor_one,tensor)
			tensor = np.einsum("abcdecbh->adeh",tensor_one)

		if True:
			edges = 4
			tensor_one = np.einsum("aicd,ijk->ajkcd",tensor,cd3)
			tensor_one = np.einsum("ablde,ijkl->abijkde",tensor_one,tensor)
			tensor_one = np.einsum("abcdefd->abcef",tensor_one)
			tensor_one = np.einsum("abcie,ijk->abcjke",tensor_one,cd3)
			tensor_one = np.einsum("abcdif,ijkl->abcdjklf",tensor_one,tensor)
			tensor_one = np.einsum("abcdecgh->abdegh",tensor_one)
			tensor_one = np.einsum("abcdif,ijk->abcdjkf",tensor_one,cd3)
			tensor_one = np.einsum("abcdejg,ijkl->abcdeiklg",tensor_one,tensor)
			tensor_one = np.einsum("abcdefadi->bcefi",tensor_one)
			tensor = np.einsum("abcij,ijk->abck",tensor_one,cd3)





		Z[i] = tensor.max()
		tensor = tensor/Z[i]

		xxx = np.einsum("abab",tensor)
		if xxx == 0:
			#print("xxx",i)
			break

		lnZ = np.log(xxx)
		lnZ /= (edges**i)
		logZ_powers_sum = sum(sorted([log(Z[j]) / (edges**j) for j in range(0,i+1)]))
		lnZ += logZ_powers_sum

		lnZ_list.append(lnZ)
		if len(lnZ_list) > 3 and abs(lnZ_list[-1] - lnZ_list[-2]) < 1e-9 and abs(lnZ_list[-1] - lnZ_list[-3]) < 1e-9:
			#print("lnZ",i)
			break
	return lnZ_list[-1]/(1./(constant*temp))*2.0

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
for muu in np.arange(-10.0,10,0.1):
	interactions = [0.0,0.0,0.0]
	method = TRG
	size = 64
	print(muu,coverage(method,'hard_triangle2',size, temperature, muu/10.0, interactions))#,heat_capasity(method,'BC_ising',size, T, muu, interactions))
	#print(temperature,heat_capasity(method,'hard_triangle',size, temperature, muu, interactions))
