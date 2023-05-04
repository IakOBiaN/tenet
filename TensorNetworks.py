from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import gc

constant = 1.

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def tensor_svd(tensor, legs_left, legs_right, chi_number = None):
	old_shape_left = [tensor.shape[i] for i in legs_left]
	old_shape_right = [tensor.shape[i] for i in legs_right]
	left = np.prod(old_shape_left)
	right = np.prod(old_shape_right)
	matrix = np.moveaxis(tensor, list(legs_left) + list(legs_right), list(range(len(tensor.shape)))).reshape(left, right)

	if chi_number is None:
		chi_number = min(matrix.shape)

	if chi_number <= min(matrix.shape)-1:
		U, S, V = scipy.sparse.linalg.svds(matrix, k = chi_number)
	else:
		U, S, V = scipy.linalg.svd(matrix, full_matrices = False, check_finite = False)

	cut = len([1 for s in S if s > 1e-12])
	if cut < 1:
		cut = 1
	if cut > chi_number:
		cut = chi_number

	sort = np.argsort(S)[-cut:]

	U = U[:, sort]
	S = S[sort]
	V = V[sort, :]

	return U.reshape(old_shape_left + [cut]), S, V.reshape([cut] + old_shape_right)

def build_tensor(calc, matrixes):
	leg_size = matrixes[0].shape[0]
	lattice = calc.lattice
	gen_tensor = calc.gen_tensor
	if lattice == "square":
		if (gen_tensor == "fusion_rule"):
			calc.nodes = 2.0
			tensor = np.einsum("ab, ibk -> iak",matrixes[0], identity(3, leg_size))
			tensor = list((np.einsum("abc, bef, ejk, jcn -> afkn", tensor, tensor, tensor, tensor), ))
		elif (gen_tensor == "svd"):
			U1, S1, V1 = tensor_svd(matrixes[0], [0], [1])
			S1 = np.sqrt(S1)
			U1 = np.einsum("ai,i->ai", U1, S1)
			V1 = np.einsum("ib,i->ib", V1, S1)
			U2, S2, V2 = tensor_svd(matrixes[1], [0], [1])
			S2 = np.sqrt(S2)
			U2 = np.einsum("ai,i->ai", U2, S2)
			V2 = np.einsum("ib,i->ib", V2, S2)
			tensor = np.einsum("ijkl, kc, ai, bj, ld -> abcd", identity(4, leg_size), U1, V1, V2, U2)
			tensor = list((tensor, ))
		elif (gen_tensor == "default"):
			tensor = np.einsum("ijkl, kc, ld -> ijcd", identity(4, leg_size), matrixes[0], matrixes[1])
			tensor = list((tensor, ))
		else:
			print("ERROR: gen_tensor parameter is incorrect!")
	elif (lattice == "triangular"):
		if (gen_tensor == "default"):
			tensor1 = np.einsum("ab,ajk->bjk",matrixes[0], identity(3, leg_size))
			tensor2 = np.einsum("ab,bjk->ajk",matrixes[1], identity(3, leg_size))
			tensor3 = np.einsum("ab,iak->ibk",matrixes[2], identity(3, leg_size))
			tensor = np.einsum("abc, bef, iea -> cfi",tensor1, tensor2, tensor3)
			tensor = list((np.einsum("abi, ijk -> abjk", tensor, identity(3, leg_size)), ))
		elif (gen_tensor == "IRF"):
			id4 = identity(4, leg_size)
			id3 = identity(3, leg_size)

			tensor = np.einsum("abcd, bi, nd, cg -> aign", id4, matrixes[1] ** 0.5, matrixes[0] ** 0.5, matrixes[2])
			tensor = np.einsum("ijkl, ajc, xc -> iaxkl", tensor, id3, matrixes[0] ** 0.5)
			tensor = np.einsum("ijklm, ablk, xb -> ijaxm", tensor, id4, matrixes[1] ** 0.5)
			tensor = np.einsum("ijklm, aml -> ijka", tensor, id3)

			U, S, V = tensor_svd(tensor, [0, 1], [2, 3])
			S = np.sqrt(S)
			U = np.einsum("abi,i->abi", U, S)
			V = np.einsum("ibc,i->ibc",V, S)
			tensor_1 = np.tensordot(id3, V, ([1], [2]))
			tensor_2 = np.tensordot(U, id3, ([1], [1]))
			tensor = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
			tensor = np.swapaxes(tensor, 2, 3)
			tensor = list((tensor, ))
	elif (lattice == "hexagonal"):
		if (gen_tensor == "default"):
			U_1, S_1, V_1 = scipy.linalg.svd(matrixes[0])
			U_2, S_2, V_2 = scipy.linalg.svd(matrixes[1])
			S_1 = np.sqrt(S_1)
			U_1 = np.einsum("ai,i->ai", U_1, S_1)
			V_1 = np.einsum("ic,i->ic", V_1, S_1)
			S_2 = np.sqrt(S_2)
			U_2 = np.einsum("ai,i->ai", U_2, S_2)
			V_2 = np.einsum("ic,i->ic", V_2, S_2)
			tensor1 = np.einsum("abc, ia, jb, cn -> ijn", identity(3, leg_size), V_1, V_2, U_1)
			tensor2 = np.einsum("abc, ia, bj, cn -> ijn", identity(3, leg_size), V_1, U_2, U_1)
			tensor = list((tensor1, tensor2))
		elif (gen_tensor == "to_square"):
			tensor = np.einsum("abc, ia, bl, cn -> iln", identity(3, leg_size), matrixes[0], matrixes[0], matrixes[1])
			tensor = list((np.einsum("abc, bjk -> ajkc", tensor, identity(3, leg_size)), ))
	elif (lattice == "complex"):
		if (gen_tensor == "default"):
			if len(matrixes) < 4:
				matrixes = matrixes + [matrixes[0]] * (4 - len(matrixes))
			id3 = identity(3, leg_size)
			ten_one = np.einsum("aix,ij,xy->ajy",id3, matrixes[0], matrixes[1])
			ten_two = np.einsum("aix,ij,xy->ajy",id3, matrixes[2], matrixes[3])
			tensor = np.einsum("kbcdef,ijk->ibjcdef",identity(6, leg_size), ten_one)
			tensor = np.einsum("abckefg,ijk->abciejgf",tensor, ten_two).reshape(leg_size**2, leg_size**2, leg_size**2, leg_size**2)
			tensor = list((tensor, ))
		elif (gen_tensor == "to_square"):
			if len(matrixes) < 4:
				matrixes = matrixes + [matrixes[0]]*(4-len(matrixes))
			id3 = identity(3, leg_size)
			ten_one = np.einsum("aix,ij,xy->ajy",id3, matrixes[2], matrixes[3])
			ten_two = np.einsum("abi,ij->abj",id3, matrixes[1])
			tensor = np.einsum("abj,ijk->abki",ten_one, ten_two)
			tensor = np.einsum("aijd,ijk->akd",tensor, identity(3, leg_size))
			ten_one = np.einsum("abi, ij->abj",identity(3, leg_size), matrixes[0])
			tensor = np.einsum("akc, ijk->icja",ten_one, tensor)

			U_1,S_1,V_1 = tensor_svd(tensor, [0, 1], [2, 3])
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
	gc.collect()
	return tensor

def hotrg_square(tensor, scale, calc):
	hotensor = tensor[0]
	size = hotensor.shape
	hotensor = np.einsum("abcd, cjkl -> abjkdl", hotensor, hotensor).reshape(size[0], size[1] ** 2, size[2], size[3] ** 2)
	if size[3] ** 2 > calc.metParam:
		simplesvd = np.tensordot(hotensor, hotensor, ([0, 1, 2], [0, 1, 2]))
		_, S1, V1 = tensor_svd(simplesvd, [0], [1])
		V1 = np.einsum("ib, i -> ib", V1, np.sqrt(S1))
		simplesvd = np.tensordot(hotensor, hotensor, ([0, 2, 3], [0, 2, 3]))
		_, S2, V2 = tensor_svd(simplesvd, [0], [1])
		V2 = np.einsum("ib, i -> ib", V2, np.sqrt(S2))
		V1V2 = np.einsum("ab, cb -> ac", V1, V2)
		U, S, V = tensor_svd(V1V2, [0], [1], calc.metParam)
		S = 1 / np.sqrt(S)
		P1 = np.einsum("ab, ca -> bc", V2, V)
		P1 = np.einsum("ab, b -> ab", P1, S)
		P2 = np.einsum("ab, ac -> bc", V1, U)
		P2 = np.einsum("ab, b -> ab", P2, S)
		hotensor = np.einsum("abci, ij -> abcj", hotensor, P1)
		hotensor = np.einsum("aicd, ij -> ajcd", hotensor, P2)

	tensor[0] = np.einsum("abcd -> dabc", hotensor)

	scale *= 2
	gc.collect()
	return tensor, scale

def trg_square(tensor, scale, calc):

	U_1,S_1,V_1 = tensor_svd(tensor[0], [0, 1], [2, 3], calc.metParam)
	U_2,S_2,V_2 = tensor_svd(tensor[0], [0, 3], [1, 2], calc.metParam)

	S_1 = np.sqrt(S_1)
	U_1 = np.einsum("abi, i -> abi", U_1, S_1)
	V_1 = np.einsum("ibc, i -> ibc", V_1, S_1)
	S_2 = np.sqrt(S_2)
	U_2 = np.einsum("abi, i -> abi", U_2, S_2)
	V_2 = np.einsum("ibc, i -> ibc",V_2, S_2)
	tensor_1 = np.tensordot(V_2, V_1, ([1], [2]))
	tensor_2 = np.tensordot(U_1, U_2, ([1], [1]))
	tensor[0] = np.tensordot(tensor_1, tensor_2, ([1, 3], [0, 2]))
	tensor[0] = np.swapaxes(tensor[0], 2, 3)

	scale *= 2
	gc.collect()
	return tensor, scale

def trg_hexagonal(tensors, scale, calc):
	tensor1 = np.einsum("abc, cjk -> abkj", tensors[0], tensors[1])
	tensor2 = np.einsum("abc, ibk -> aikc", tensors[0], tensors[1])
	tensor3 = np.einsum("abc, ija -> ibcj", tensors[0], tensors[1])

	U_1,S_1,V_1 = tensor_svd(tensor1, [1, 2], [3, 0], calc.metParam)
	U_2,S_2,V_2 = tensor_svd(tensor2, [0, 1], [2, 3], calc.metParam)
	U_3,S_3,V_3 = tensor_svd(tensor3, [0, 1], [2, 3], calc.metParam)

	S_1 = np.sqrt(S_1)
	U_1 = np.einsum("abi, i -> abi", U_1, S_1)
	V_1 = np.einsum("ibc, i -> ibc", V_1, S_1)
	S_2 = np.sqrt(S_2)
	U_2 = np.einsum("abi, i -> abi", U_2, S_2)
	V_2 = np.einsum("ibc, i -> ibc", V_2, S_2)
	S_3 = np.sqrt(S_3)
	U_3 = np.einsum("abi, i -> abi", U_3, S_3)
	V_3 = np.einsum("ibc, i -> ibc", V_3, S_3)

	tensor1 = np.einsum("abc, deb, cek -> adk", V_2, V_1, U_3)
	tensor2 = np.einsum("abc, cef, ebk -> afk", V_3, U_1, U_2)

	scale *= 3.0
	return list((tensor1, tensor2)), scale

def hotrg_step(tensors, scale, norm, calc):
	norm = tensors[0].max()
	if norm != 0:
		for i, ten in enumerate(tensors):
			tensors[i] = ten/norm
		scale += np.log(norm)
	tensors, scale = hotrg_square(tensors, scale, calc)
	tensors, scale = hotrg_square(tensors, scale, calc)
	norm = np.einsum("abab->",tensors[0])
	if norm < 0:
		norm = -norm
	return (tensors, scale, norm)

def trg_step(tensors, scale, norm, calc):
	if (calc.metModification == "hex"):
		calc.scale = 9
		if len(tensors) < 2:
			U, S, V = tensor_svd(tensors[0], [0, 1], [3, 2])
			S = np.sqrt(S)
			U = np.einsum("abi, i -> abi", U, S)
			V = np.einsum("ibc, i -> ibc", V, S)
			tensors = list((U, V))
		norm = abs(np.einsum("abc, cba -> ", tensors[0], tensors[1]))
		if norm != 0:
			for i, ten in enumerate(tensors):
				tensors[i] = ten/sqrt(norm)
			scale += np.log(norm)
		tensors, scale = trg_hexagonal(tensors, scale, calc)
		tensors, scale = trg_hexagonal(tensors, scale, calc)
		norm = abs(np.einsum("abc, cba -> ", tensors[0], tensors[1]))
		if norm < 0:
			norm = -norm
	else:
		norm = tensors[0].max()
		if norm != 0:
			for i, ten in enumerate(tensors):
				tensors[i] = ten/norm
			scale += np.log(norm)
		tensors, scale = trg_square(tensors, scale, calc)
		tensors, scale = trg_square(tensors, scale, calc)
		norm = np.einsum("abab->",tensors[0])
		if norm < 0:
			norm = -norm
	gc.collect()
	return (tensors, scale, norm)

def htn_step(tensors, scale, norm, calc):
	size = calc.metParam
	edges = (1 + size * 2) ** 2
	calc.scale = edges
	calc.coord = 4
	calc.nodes = 0.5
	tensor = tensors[0]
	norm = tensor.max()
	if norm != 0:
		tensor /= norm
		scale += np.log(norm)

	cd3 = identity(3, tensor.shape[0])
	cd4 = identity(4, tensor.shape[0])

	dop_tensor = identity(size + 2, tensor.shape[0])
	dop_tensor_2 = identity(size + 2, tensor.shape[0])

	doubled_tensor = np.einsum("ij, aic -> ajc", tensor, cd3)
	doubled_tensor = np.einsum("ij, abi -> abj", tensor, doubled_tensor)
	doubled_tensor = np.einsum("ijk, ajk -> ai", doubled_tensor, cd3)

	for _ in range(size):
		dop_tensor = np.tensordot(dop_tensor, doubled_tensor, axes = ([1], [0]))
	dop_tensor = np.tensordot(dop_tensor, tensor, axes=([1], [0]))

	for _ in range(size + (size - 1)):
		dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-1], [0]))
		for j in range(size):
			dop_tensor = np.tensordot(dop_tensor, cd3, axes=([-2 - j], [0]))
			dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-2], [0]))
			dop_tensor = np.tensordot(dop_tensor, cd3, axes=([-1, -3], [0, 1]))
			dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-2], [0]))

	for _ in range(size):
		dop_tensor = np.tensordot(dop_tensor, doubled_tensor, axes=([1], [0]))
	dop_tensor = np.tensordot(dop_tensor, tensor, axes=([1], [0]))
	tensor = np.tensordot(dop_tensor, dop_tensor_2, axes=(np.arange(1, size + 2, 1), np.arange(1, size + 2, 1)))
	tensors[0] = tensor
	norm = 1
	scale *= edges
	gc.collect()
	return (tensors, scale, norm)
