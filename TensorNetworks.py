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
	elif (lattice == "tr_to_sqr"):
		tensor = np.einsum("abc, dfb, icf -> adi",tensor,tensor,tensor)
		tensor = list((np.einsum("abi, ijk -> abjk", tensor, identity(3, leg_size)), ))
	elif (lattice == "tr_to_sqr_susmost"):
		id4 = identity(4, leg_size)
		id3 = identity(3, leg_size)
		tensor = np.einsum("abcd, efgh, ijk, mno, bi, kf, dn, oh, cg -> ajem", id4, id4, id3, id3, matrixes[0] ** 0.5, matrixes[0] ** 0.5, matrixes[0] ** 0.5, matrixes[0] ** 0.5, matrixes[0])
		U, S, V = split_by_svd(tensor, [0, 1], [2, 3])
		S = np.sqrt(S)
		U = np.einsum("abi,i->abi", U, S)
		V = np.einsum("ibc,i->ibc",V, S)
		tensor_1 = np.tensordot(id3, V, ([1], [2]))
		tensor_2 = np.tensordot(U, id3, ([1], [1]))
		tensor = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
		tensor = np.swapaxes(tensor, 2, 3)
		tensor = list((tensor, ))
	elif (lattice == "hex"):
		U_1, S_1, V_1 = scipy.linalg.svd(matrixes[0])
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
		tensor = list((tensor1, tensor2))
	elif (lattice == "hex_to_sqr"):
		tensor = np.einsum("abc, ia, bl, cn -> iln", identity(3, leg_size), matrixes[0], matrixes[0], matrixes[1])
		tensor = list((np.einsum("abc, bjk -> ajkc", tensor, identity(3, leg_size)), ))
	elif (lattice == "complex_to_sqr2"):
		if len(matrixes) < 4:
			matrixes = matrixes + [matrixes[0]]*(4-len(matrixes))
		id3 = identity(3, leg_size)
		ten_one = np.einsum("aix,ij,xy->ajy",id3, matrixes[0], matrixes[1])
		ten_two = np.einsum("aix,ij,xy->ajy",id3, matrixes[2], matrixes[3])
		tensor = np.einsum("kbcdef,ijk->ibjcdef",identity(6, leg_size), ten_one)
		tensor = np.einsum("abckefg,ijk->abciejgf",tensor, ten_two).reshape(leg_size**2, leg_size**2, leg_size**2, leg_size**2)
		tensor = list((tensor, ))
	elif (lattice == "complex_to_sqr"):
		if len(matrixes) < 4:
			matrixes = matrixes + [matrixes[0]]*(4-len(matrixes))
		id3 = identity(3, leg_size)
		ten_one = np.einsum("aix,ij,xy->ajy",id3, matrixes[2], matrixes[3])
		ten_two = np.einsum("abi,ij->abj",id3, matrixes[1])
		tensor = np.einsum("abj,ijk->abki",ten_one, ten_two)
		tensor = np.einsum("aijd,ijk->akd",tensor, identity(3, leg_size))
		ten_one = np.einsum("abi, ij->abj",identity(3, leg_size), matrixes[0])
		tensor = np.einsum("akc, ijk->icja",ten_one, tensor)

		U_1,S_1,V_1 = split_by_svd(tensor, [0, 1], [2, 3])
		U_2 = identity(3, leg_size)
		V_2 = identity(3, leg_size)

		S_1 = np.sqrt(S_1)
		U_1 = np.einsum("abi,i->abi", U_1, S_1)
		V_1 = np.einsum("ibc,i->ibc",V_1, S_1)

		ten_one = np.tensordot(V_2, V_1, ([1], [2]))
		ten_two = np.tensordot(U_1, U_2, ([1], [1]))
		tensor = np.tensordot(ten_one, ten_two, ([1, 3],[0, 2]))

		tensor = np.swapaxes(tensor, 2, 3)
		tensor = list((tensor, ))
	return tensor

def hotrg_square(tensor, scale, chi_number = 64, chi_min = 1e-8):
	hotensor = tensor[0]
	size = hotensor.shape
	hotensor = np.einsum("abcd,cjkl->abjkdl", hotensor, hotensor).reshape(size[0],size[1]*size[1],size[2],size[3]*size[3])

	if size[3]*size[3] > chi_number:
		U,S,V = split_by_svd(hotensor, [0, 1, 2], [3], chi_min, chi_number)
		S = np.sqrt(S)
		U = np.einsum("abci,i->abci", U, S)
		V = np.einsum("ib,i->ib",V, S)
		hotensor = np.einsum("ajcd,ij->aicd", U, V)

	tensor[0] = np.einsum("abcd->dabc", hotensor)

	scale *= 2
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

def hotrg_step(tensor, scale, chi_number = 64, chi_min = 1e-8, lattice = "square"):
	norm = tensor[0].max()
	#print(np.log(norm))
	if norm != 0:
		for i, ten in enumerate(tensor):
			tensor[i] = ten/norm
		scale += np.log(norm)
	tensor, scale = hotrg_square(tensor, scale, chi_number, chi_min)
	tensor, scale = hotrg_square(tensor, scale, chi_number, chi_min)
	return (tensor, scale)

def trg_step(tensor, scale, chi_number = 64, chi_min = 1e-8, lattice = "square"):

	if (lattice == "hex"):
		norm = abs(np.einsum("abc, cba -> ", tensor[0], tensor[1]))
		if norm != 0:
			for i, ten in enumerate(tensor):
				tensor[i] = ten/sqrt(norm)
			scale += np.log(norm)
		tensor, scale = trg_hexagonal(tensor, scale, chi_number, chi_min)
		tensor, scale = trg_hexagonal(tensor, scale, chi_number, chi_min)
	else:
		norm = tensor[0].max()
		if norm != 0:
			for i, ten in enumerate(tensor):
				tensor[i] = ten/norm
			scale += np.log(norm)
		tensor, scale = trg_square(tensor, scale, chi_number, chi_min)
		tensor, scale = trg_square(tensor, scale, chi_number, chi_min)

	return (tensor, scale)
