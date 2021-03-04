from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm

constant = 1.

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def matricise(tensor, legs_one, legs_two):
	one = np.prod([tensor.shape[i] for i in legs_one])
	two = np.prod([tensor.shape[i] for i in legs_two])
	return np.moveaxis(tensor, list(legs_one) + list(legs_two), list(range(len(tensor.shape)))).reshape(one, two)

def split_by_svd(tensor, legs_one, legs_two, chi_min = 0.0, chi_number = None):
	if legs_one is not None:
		legs_one = [i % len(tensor.shape) for i in legs_one]
	if legs_two is not None:
		legs_two = [i % len(tensor.shape) for i in legs_two]

	if legs_two is None:
		legs_two = sorted(list(set(range(len(tensor.shape))) - set(legs_one)))
	elif legs_one is None:
		legs_one = sorted(list(set(range(len(tensor.shape))) - set(legs_two)))

	shape_one = [tensor.shape[i] for i in legs_one]
	shape_two = [tensor.shape[i] for i in legs_two]
	matrix = matricise(tensor, legs_one, legs_two)

	if chi_number is not None and chi_number <= min(matrix.shape)-1:
		U, S, V = scipy.sparse.linalg.svds(matrix, k = chi_number)
	else:
		U, S, V = scipy.linalg.svd(matrix, full_matrices = False, check_finite = False)

	cut = len([1 for s in S if s > chi_min])
	if cut < 1:
		cut = 1
	if chi_number is not None and cut > chi_number:
		cut = chi_number

	sort = np.argsort(S)[-cut:]

	U = U[:, sort]
	S = S[sort]
	V = V[sort, :]

	return U.reshape(shape_one+ [cut]), S, V.reshape([cut] + shape_two)

def build_tensor(matrixes, lattice = "square"):
	leg_size = matrixes[0].shape[0]
	tensor = np.einsum("ab,ibk->iak",matrixes[0], identity(3, leg_size))
	if (lattice == "square"):
		tensor = list((np.einsum("abc, bef, ejk, jcn -> afkn", tensor, tensor, tensor, tensor), ))
	elif (lattice == "triangle"):
		tensor = np.einsum("abc, dfb, icf -> adi",tensor,tensor,tensor)
		tensor = list((np.einsum("abi, ijk -> abjk", tensor, identity(3, leg_size)), ))
	elif (lattice == "hexagonal"):
		"""U_1, S_1, V_1 = scipy.linalg.svd(matrixes[0])
		U_2, S_2, V_2 = scipy.linalg.svd(matrixes[1])
		S_1 = np.sqrt(S_1)
		U_1 = np.einsum("ai,i->ai", U_1, S_1)
		V_1 = np.einsum("ic,i->ic", V_1, S_1)
		S_2 = np.sqrt(S_2)
		U_2 = np.einsum("ai,i->ai", U_2, S_2)
		V_2 = np.einsum("ic,i->ic", V_2, S_2)
		#U_1 = sqrtm(matrixes[0])
		#V_1 = U_1
		#U_2 = sqrtm(matrixes[1])
		#V_2 = U_2
		tensor1 = np.einsum("abc, ia, jb, cn -> ijn", identity(3, leg_size), V_1, V_2, U_1)
		tensor2 = np.einsum("abc, ia, bj, cn -> ijn", identity(3, leg_size), V_1, U_2, U_1)
		tensor = list((tensor1, tensor2))"""

		tensor = np.einsum("abc, ia, bl, cn -> iln", identity(3, leg_size), matrixes[0], matrixes[0], matrixes[1])
		tensor = list((np.einsum("abc, bjk -> ajkc", tensor, identity(3, leg_size)), ))
	return tensor

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
	two2 = np.array([np.exp(line) for line in two])
	three *= 1./(constant*temp)
	three = np.array([np.exp(line) for line in three])
	cd = identity(3, one.shape[0])
	one = np.einsum("ab,iak->ibk",one,cd)
	two = np.einsum("ab,iak->ibk",three,cd)
	three = np.einsum("ab,ibk->iak",two2,cd)
	tensor = np.einsum("abc,deb,ice->adi",one,two,three)
	tensor = np.einsum("abi,ijk->abjk", tensor,cd)
	tensor = list((tensor, ))
	return tensor

def hotrg_square(tensor, scale, chi_number = 64, chi_min = 1e-8):
	
	return tensor, scale

def trg_square(tensor, scale, chi_number = 64, chi_min = 1e-8):

	U_1,S_1,V_1 = split_by_svd(tensor[0], [0, 1], [2, 3], chi_min, chi_number)
	U_2,S_2,V_2 = split_by_svd(tensor[0], [0, 3], [1, 2], chi_min, chi_number)

	S_1 = np.sqrt(S_1)
	U_1 = np.einsum("abi,i->abi", U_1, S_1)
	V_1 = np.einsum("ibc,i->ibc",V_1, S_1)
	S_2 = np.sqrt(S_2)
	U_2 = np.einsum("abi,i->abi", U_2, S_2)
	V_2 = np.einsum("ibc,i->ibc",V_2, S_2)
	tensor_1 = np.tensordot(V_2, V_1, ([1], [2]))
	tensor_2 = np.tensordot(U_1, U_2, ([1], [1]))
	tensor[0] = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
	tensor[0] = np.swapaxes(tensor[0], 2, 3)

	scale *= 2
	return tensor, scale

def trg_hexagonal(tensors, scale, chi_number = 64, chi_min = 1e-8):
	tensor1 = np.einsum("abc, cjk -> abkj", tensors[0], tensors[1])
	tensor2 = np.einsum("abc, ibk -> aikc", tensors[0], tensors[1])
	tensor3 = np.einsum("abc, ija -> ibcj", tensors[0], tensors[1])

	U_1,S_1,V_1 = split_by_svd(tensor1, [1, 2], [3, 0], chi_min, chi_number)
	U_2,S_2,V_2 = split_by_svd(tensor2, [0, 1], [2, 3], chi_min, chi_number)
	U_3,S_3,V_3 = split_by_svd(tensor3, [0, 1], [2, 3], chi_min, chi_number)

	S_1 = np.sqrt(S_1)
	U_1 = np.einsum("abi,i->abi", U_1, S_1)
	V_1 = np.einsum("ibc,i->ibc",V_1, S_1)
	S_2 = np.sqrt(S_2)
	U_2 = np.einsum("abi,i->abi", U_2, S_2)
	V_2 = np.einsum("ibc,i->ibc",V_2, S_2)
	S_3 = np.sqrt(S_3)
	U_3 = np.einsum("abi,i->abi", U_3, S_3)
	V_3 = np.einsum("ibc,i->ibc",V_3, S_3)

	tensor1 = np.einsum("abc, deb, cek -> adk", V_2, V_1, U_3)
	tensor2 = np.einsum("abc, cef, ebk -> afk", V_3, U_1, U_2)

	scale *= 3.0
	return list((tensor1, tensor2)), scale

def trg_step(tensor, scale, chi_number = 64, chi_min = 1e-8, lattice = "square"):

	if (lattice == "square"):
		norm = tensor[0].max()
		if norm != 0:
			for i, ten in enumerate(tensor):
				tensor[i] = ten/norm
			scale += np.log(norm)
		tensor, scale = trg_square(tensor, scale, chi_number, chi_min)
		tensor, scale = trg_square(tensor, scale, chi_number, chi_min)
	elif (lattice == "hexagonal"):
		norm = abs(np.einsum("abc, cba -> ", tensor[0], tensor[1]))
		if norm != 0:
			for i, ten in enumerate(tensor):
				tensor[i] = ten/sqrt(norm)
			scale += np.log(norm)
		tensor, scale = trg_hexagonal(tensor, scale, chi_number, chi_min)
		tensor, scale = trg_hexagonal(tensor, scale, chi_number, chi_min)
	return (tensor, scale)
