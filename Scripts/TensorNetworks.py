from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import gc

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
			U1, S1, V1 = tensor_svd(matrixes[0], [0], [1], chi_number = calc.metParam)
			S1 = np.sqrt(S1)
			U1 = np.einsum("ai,i->ai", U1, S1)
			V1 = np.einsum("ib,i->ib", V1, S1)
			U2, S2, V2 = tensor_svd(matrixes[1], [0], [1], chi_number = calc.metParam)
			S2 = np.sqrt(S2)
			U2 = np.einsum("ai,i->ai", U2, S2)
			V2 = np.einsum("ib,i->ib", V2, S2)
			"""tensor = np.einsum("ijkl, kc -> ijcl", identity(4, leg_size), U1)
			tensor = np.einsum("ijkl, ai -> ajkl", tensor, V1)
			tensor = np.einsum("ijkl, bj -> ibkl", tensor, V2)
			tensor = np.einsum("ijkl, ld -> ijkd", tensor, U2)"""
			tensor_1 = np.einsum("ijk, jc -> ick", identity(3, leg_size), U1)
			tensor_2 = np.einsum("ijk, ai -> ajk", identity(3, leg_size), V1)
			tensor_1 = np.einsum("ijk, bi -> bjk", tensor_1, V2)
			tensor_2 = np.einsum("ijk, jd -> idk", tensor_2, U2)
			tensor = np.einsum("ijk, abk -> aijb", tensor_1, tensor_2)
			tensor = list((tensor, ))
		elif (gen_tensor == "default"):
			id4 = identity(4, leg_size)
			tensor = np.einsum("ijkl, kc -> ijcl", id4, matrixes[0])
			tensor = np.einsum("ijkl, ld -> ijkd", tensor, matrixes[1])
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
		elif (gen_tensor == "default_m"):
			tensor = np.einsum("ijkl, jp, koz, mnop -> imnzl",identity(4, leg_size), matrixes[0], matrixes[-1], identity(4, leg_size))
			tensor = np.einsum("ijklm, kr, my, xylr -> ijx", tensor, matrixes[0], matrixes[0], identity(4, leg_size))
			tensor = np.einsum("ijk, aiz, dxk, xyz -> ajdy", tensor, identity(3, leg_size), identity(3, leg_size), matrixes[-1])

			U, S, V = tensor_svd(tensor, [0, 1], [2, 3])
			S = np.sqrt(S)
			U = np.einsum("abi,i->abi", U, S)
			V = np.einsum("ibc,i->ibc",V, S)
			tensor_1 = np.tensordot(identity(3, leg_size), V, ([1], [2]))
			tensor_2 = np.tensordot(U, identity(3, leg_size), ([1], [1]))
			tensor = np.tensordot(tensor_1, tensor_2, ([1, 3],[0, 2]))
			tensor = np.swapaxes(tensor, 2, 3)
			tensor = list((tensor, ))
		elif (gen_tensor == "IRF_m"):
			tensor = np.einsum("ijklmn, pn, xmy, apxb -> ijklyba", identity(6, leg_size), matrixes[0] ** 0.5, matrixes[-1], identity(4, leg_size))
			tensor = np.einsum("ijklabc, jz, kyd, rxyz -> irxdlabc", tensor, matrixes[0] ** 0.5, matrixes[-1], identity(4, leg_size))
			tensor = np.einsum("ijklabcd, xk, ay, cz, ozbylx -> ijod", tensor, matrixes[0] ** 0.5, matrixes[0], matrixes[0] ** 0.5, identity(6, leg_size))

			U, S, V = tensor_svd(tensor, [0, 1], [2, 3])
			S = np.sqrt(S)
			U = np.einsum("abi,i->abi", U, S)
			V = np.einsum("ibc,i->ibc", V, S)
			tensor_1 = np.tensordot(identity(3, leg_size), V, ([1], [2]))
			tensor_2 = np.tensordot(U, identity(3, leg_size), ([1], [1]))
			tensor = np.tensordot(tensor_1, tensor_2, ([1, 3], [0, 2]))
			tensor = np.swapaxes(tensor, 2, 3)
			tensor = list((tensor,))
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
		elif (gen_tensor == "six_leg_tensor"):
			tensor = matrixes[0]
			U1, S1, V1 = tensor_svd(tensor, [0, 1], [2, 3, 4, 5], chi_number = calc.metParam)
			S1 = np.sqrt(S1)
			U1 = np.einsum("abi,i->abi", U1, S1)
			V1 = np.einsum("icdef,i->cdefi",V1, S1)

			U2, S2, V2 = tensor_svd(V1, [0, 1], [2, 3, 4], chi_number = calc.metParam)
			S2 = np.sqrt(S2)
			U2 = np.einsum("cdi,i->cdi", U2, S2)
			V2 = np.einsum("iefx,i->efxi",V2, S2)

			U3, S3, V3 = tensor_svd(V2, [0, 1], [2, 3], chi_number = calc.metParam)
			S3 = np.sqrt(S3)
			U3 = np.einsum("efi,i->efi", U3, S3)
			V3 = np.einsum("ixz,i->ixz",V3, S3)

			tensor_temp = np.einsum("abc, iak, bin -> knc", U1, U2, U3)

			tensor = np.einsum("abc, bjk -> ajkc", tensor_temp, V3)

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
	else:
		tensor = list((matrixes[0], ))

	if calc.join_tensors[0] > 1:
		temp_ten = tensor[0]
		for _ in range(calc.join_tensors[0] - 1):
			ten = tensor[0]

			temp_sh = temp_ten.shape
			t_sh = ten.shape
			temp_ten = np.einsum("ijkl, kbcd -> ijbcld", temp_ten, ten).reshape(temp_sh[0], temp_sh[1] * t_sh[1], temp_sh[2], temp_sh[3] * t_sh[3])

		tensor = list((temp_ten, ))

	if calc.join_tensors[1] > 1:
		temp_ten = tensor[0]
		for _ in range(calc.join_tensors[1] - 1):
			ten = tensor[0]

			temp_sh = temp_ten.shape
			t_sh = ten.shape
			temp_ten = np.einsum("ijkl, alcd -> aijckd", temp_ten, ten).reshape(temp_sh[0] * t_sh[0], temp_sh[1], temp_sh[2] * t_sh[2], temp_sh[3])

		tensor = list((temp_ten, ))

	if calc.method == "tm":
		ten = tensor[0]
		if calc.metParam > 1:
			ten_dop = ten
			t_sh = ten.shape
			for i in range(calc.metParam - 1):
				sh = ten_dop.shape
				ten_dop = np.einsum("ijkl, alcd -> aijkcd", ten_dop, ten).reshape(sh[0] * t_sh[0], sh[1], sh[2] * t_sh[0], sh[3])

			if calc.metModification != "default":
				tmXhi = calc.metModification[1]
				for i in range(calc.metModification[0] - 1):
					if ten_dop.shape[0] > tmXhi:
						#horizontal tesor reduction
						ten_dop = np.einsum("ijkl->lijk", ten_dop)
						simplesvd = np.tensordot(ten_dop, ten_dop, ([0, 1, 2], [0, 1, 2]))
						_, S1, V1 = tensor_svd(simplesvd, [0], [1])
						V1 = np.einsum("ib, i -> ib", V1, np.sqrt(S1))
						simplesvd = np.tensordot(ten_dop, ten_dop, ([0, 2, 3], [0, 2, 3]))
						_, S2, V2 = tensor_svd(simplesvd, [0], [1])
						V2 = np.einsum("ib, i -> ib", V2, np.sqrt(S2))
						V1V2 = np.einsum("ab, cb -> ac", V1, V2)
						U, S, V = tensor_svd(V1V2, [0], [1], tmXhi)
						S = 1 / np.sqrt(S)
						P1 = np.einsum("ab, ca -> bc", V2, V)
						P1 = np.einsum("ab, b -> ab", P1, S)
						P2 = np.einsum("ab, ac -> bc", V1, U)
						P2 = np.einsum("ab, b -> ab", P2, S)
						ten_dop = np.einsum("abci, ij -> abcj", ten_dop, P1)
						ten_dop = np.einsum("aicd, ij -> ajcd", ten_dop, P2)
						ten_dop = np.einsum("lijk->ijkl", ten_dop)
					sh = ten_dop.shape
					ten_dop = np.einsum("ijkl, alcd -> aijkcd", ten_dop, ten_dop).reshape(sh[0] ** 2, sh[1], sh[2] ** 2, sh[3])

			ten = ten_dop
		ten = np.einsum("abcb->ac", ten)
		tensor = list((ten, ))


	gc.collect()
	return tensor

def tm_step(tensors, scale, norm, calc):
	norm = tensors[0].max()
	if norm != 0:
		for i, ten in enumerate(tensors):
			tensors[i] = ten / norm
		scale += np.log(norm)

	tensors[0] = np.dot(tensors[0], tensors[0])
	scale *= 2

	norm = np.einsum("aa->", tensors[0])
	if norm < 0:
		norm = -norm
	if norm == 0:
		norm = 1
	gc.collect()
	return (tensors, scale, norm)

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

def btrg_square(tensor, scale, calc):
	if calc.metModification == "default":
		k = -0.5
	else:
		k = calc.metModification

	S1_btrg = tensor[1]
	S2_btrg = tensor[2]

	U_1,S_1,V_1 = tensor_svd(tensor[0], [0, 1], [2, 3], calc.metParam)
	U_2,S_2,V_2 = tensor_svd(tensor[0], [0, 3], [1, 2], calc.metParam)

	E = np.einsum("ab, b -> ab", np.eye(len(S_1)), np.power(S_1, k))
	F = np.einsum("ab, b -> ab", np.eye(len(S_2)), np.power(S_2, k))

	S_1 = np.power(S_1, (1.0 - k) / 2.0)
	U_1 = np.einsum("abi, i -> abi", U_1, S_1)
	V_1 = np.einsum("ibc, i -> ibc", V_1, S_1)
	S_2 = np.power(S_2, (1.0 - k) / 2.0)
	U_2 = np.einsum("abi, i -> abi", U_2, S_2)
	V_2 = np.einsum("ibc, i -> ibc",V_2, S_2)

	#BTRG part
	V_1 = np.einsum("aic, ij -> ajc", V_1, S2_btrg)
	V_2 = np.einsum("ajc, ij -> aic", V_2, S1_btrg)
	U_1 = np.einsum("jbc, ij -> ibc", U_1, S2_btrg)
	U_2 = np.einsum("aic, ij -> ajc", U_2, S1_btrg)
	tensor[1] = E
	tensor[2] = F

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
	if norm == 0:
		norm = 1
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

def btrg_step(tensors, scale, norm, calc):
	if (calc.metModification == "hex"):
		print("Error! HEX now is not ready!")
		exit()
		"""calc.scale = 9
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
			norm = -norm"""
	else:
		if len(tensors) == 1:
			tensors.append(np.eye(tensors[0].shape[1]))
			tensors.append(np.eye(tensors[0].shape[0]))
		temp_tensor = np.einsum("ajcd, ij->aicd",tensors[0], tensors[1])
		temp_tensor = np.einsum("abkd, kl->abld",temp_tensor, tensors[2])
		norm = temp_tensor.max()
		if norm != 0:
			tensors[0] /= norm
			scale += np.log(norm)
		tensors, scale = btrg_square(tensors, scale, calc)
		temp_tensor = np.einsum("ajcd, ij->aicd",tensors[0], tensors[1])
		temp_tensor = np.einsum("abkd, kl->abld",temp_tensor, tensors[2])
		norm = temp_tensor.max()
		if norm != 0:
			tensors[0] /= norm
			scale += np.log(norm)
		tensors, scale = btrg_square(tensors, scale, calc)
		norm = np.einsum("ajcd, ij->aicd",tensors[0], tensors[1])
		norm = np.einsum("abkd, kl->abld",norm, tensors[2])
		norm = np.einsum("abab->",norm)
		#norm = np.einsum("abab->",tensors[0])
		if norm < 0:
			norm = -norm
	gc.collect()
	return (tensors, scale, norm)

def htn_step(tensors, scale, norm, calc):
	if calc.lattice == "FSHL":
		size = calc.metParam
		edges_in = (1 + size * 2) ** 2
		edges = edges_in
		nodes =  2 + (size * 2) * (size + 1)
		calc.nodes = nodes
		for i in range(calc.scale):
			calc.nodes += edges * (nodes - 2)
			edges *= edges_in
			#print(calc.nodes, edges)
		tensor = tensors[0]
		norm = tensor.max()
		if norm != 0:
			tensor /= norm
			scale += np.log(norm)
			scale *= edges_in

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
		tensor = np.tensordot(dop_tensor, dop_tensor_2, axes = (np.arange(1, size + 2, 1), np.arange(1, size + 2, 1)))
	elif calc.lattice == "diamond":
		edges_in = 4
		edges = edges_in
		nodes = 4
		calc.nodes = nodes
		for i in range(calc.scale):
			calc.nodes += edges * (nodes - 2)
			edges *= edges_in
		tensor = tensors[0]
		norm = tensor.max()
		if norm != 0:
			tensor /= norm
			scale += np.log(norm)
			scale *= edges_in

		cd3 = identity(3, tensor.shape[0])
		tensor_dop = np.einsum("ia, abc -> ibc", tensor, cd3)
		tensor_dop = np.einsum("ia, abc -> ibc", tensor, tensor_dop)
		tensor_dop = np.einsum("ib, abc -> aic", tensor, tensor_dop)
		tensor_dop = np.einsum("ib, abc -> aic", tensor, tensor_dop)
		tensor = np.einsum("ijk, jkc -> ic", cd3, tensor_dop)

	tensors[0] = tensor
	norm = np.einsum("ij -> ", tensor)
	gc.collect()
	return (tensors, scale, norm)
