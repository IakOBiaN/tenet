import numpy as np
import scipy as sp
import matplotlib.pyplot as pt

constant = 0.008314

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def tensor_svd(tensor, legs_left, legs_right, chi_number):
	old_shape_left = [tensor.shape[i] for i in legs_left]
	old_shape_right = [tensor.shape[i] for i in legs_right]
	left = np.prod(old_shape_left)
	right = np.prod(old_shape_right)
	matrix = np.moveaxis(tensor, list(legs_left) + list(legs_right), list(range(len(tensor.shape)))).reshape(right, right)

	if chi_number <= min(matrix.shape)-1:
		U, S, V = sp.sparse.linalg.svds(matrix, k = chi_number)
	else:
		U, S, V = sp.linalg.svd(matrix, full_matrices = False, check_finite = False)

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

def build_matrix (model, temp, m_par, lattice = "sqare"):

	neighbours = 4
	if lattice == "triangular":
		neighbours = 6
	elif lattice == "hexagonal":
		neighbours = 3
	matrix_dict = {
		"0NN" : (np.array([[0.0, m_par[0] / neighbours], [m_par[0] / neighbours, - m_par[1] + m_par[0] / (neighbours / 2.0)]]) ,) * 3,
		"1NN" : (np.array([[0.0, m_par[0] / neighbours], [m_par[0] / neighbours, -1e10 + m_par[0]]]), ) * 3
	}

	matrixes = list(matrix_dict.get(model))
	assert (matrixes is not None), "Ошибка! такой модели нет в базе данных."
	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i] / (constant * temp)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])
	return matrixes

def build_tensor(matrixes, lattice = "square"):
	leg_size = matrixes[0].shape[0]
	tensor = np.einsum("ab,ibk->iak",matrixes[0], identity(3, leg_size))
	if (lattice == "square"):
		tensor = list((np.einsum("abc, bef, ejk, jcn -> afkn", tensor, tensor, tensor, tensor), ))
	elif (lattice == "triangular"):
		tensor1 = np.einsum("ab,ajk->bjk",matrixes[0], identity(3, leg_size))
		tensor2 = np.einsum("ab,bjk->ajk",matrixes[1], identity(3, leg_size))
		tensor3 = np.einsum("ab,iak->ibk",matrixes[2], identity(3, leg_size))
		tensor = np.einsum("abc, bef, iea -> cfi", tensor1, tensor2, tensor3)
		tensor = list((np.einsum("abi, ijk -> abjk", tensor, identity(3, leg_size)), ))
	elif (lattice == "hexagonal"):
		tensor = np.einsum("abc, ia, bl, cn -> iln", identity(3, leg_size), matrixes[0], matrixes[0], matrixes[1])
		tensor = list((np.einsum("abc, bjk -> ajkc", tensor, identity(3, leg_size)), ))
	return tensor

def trg_square(tensor, scale, chi_number = 64):

	U_1, S_1, V_1 = tensor_svd(tensor[0], [0, 1], [2, 3], chi_number)
	U_2, S_2, V_2 = tensor_svd(tensor[0], [0, 3], [1, 2], chi_number)

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

def trg_step(tensor, scale, chi_number = 64, lattice = "square"):

	norm = tensor[0].max()
	if norm != 0:
		for i, ten in enumerate(tensor):
			tensor[i] = ten/norm
		scale += np.log(norm)
	tensor, scale = trg_square(tensor, scale, chi_number)
	tensor, scale = trg_square(tensor, scale, chi_number)

	return (tensor, scale)

def simulate(model = "0NN", lattice = "square", chi_number = 32, temp = 120.0, m_par = [0.0]*3):

	tensors = build_matrix(model, temp, m_par, lattice)
	tensors = build_tensor(tensors, lattice)

	scale = 0.0
	old_scale = -1.0
	nodes = 2.0
	if lattice == "triangular":
		nodes = 1.0

	i = 0
	for i in range(300):
		(tensors, scale) = trg_step(tensors, scale, chi_number, lattice)

		if abs(old_scale - scale/4.0) < 1e-8:
			break
		else:
			old_scale = scale
	nodes *= 4.0 ** (i + 1)
	norm = np.einsum("abab->",tensors[0])
	if norm < 0:
		norm = -norm
	return (scale + np.log(norm)) / (nodes / (constant * temp))

def full(model, lattice, chi_number = 32, temp = 120., m_par = [0.0] * 10, step = 0.01):
	BTP_mu = []
	BTP_temp = []
	for mu_TRG in [m_par[0] - step, m_par[0] + step]:
		lnZ = simulate(model, lattice, chi_number, temp, [mu_TRG] + m_par[1:])
		BTP_mu.append(lnZ)
	for temp_TRG in [temp - step, temp, temp + step]:
		lnZ = simulate(model, lattice, chi_number, temp_TRG, m_par)
		BTP_temp.append(lnZ)

	cov = - (BTP_mu[0] - BTP_mu[1]) / (step * 2.0)
	ent = - (BTP_temp[0] - BTP_temp[2]) / (step * 2.0)
	sus = (BTP_mu[0] - 2.0 * BTP_temp[1] + BTP_mu[1]) / (step ** 2.0)
	cap = (BTP_temp[0] - 2.0 * BTP_temp[1] + BTP_temp[2]) / (step ** 2.0)
	return cov, ent, sus, cap

def main():
	model = "1NN"
	lattice = "triangular"
	temp = 120.0
	mu = 0.0
	inter = -2.0
	chi_number = 10
	step = 0.01
	for mu in np.arange(-5.0, 8.01, 0.1):
		m_par = [mu, inter]
		result = full(model, lattice, chi_number, temp, m_par, step)
		print(mu, result[0])

if __name__ == "__main__":
    main()
