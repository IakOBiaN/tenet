import numpy as np
import scipy as sp

np.random.seed(0)
matrix = np.random.sample((5, 5))
print(sp.sparse.linalg.svds(matrix, k = 4))


def tensor_svd(tensor, legs_1, legs_2, chi_number):
	old_shape_1 = [tensor.shape[i] for i in legs_1]
	old_shape_2 = [tensor.shape[i] for i in legs_2]
	one = np.prod(old_shape_1)
	two = np.prod(old_shape_2)
	matrix = np.moveaxis(tensor, list(legs_1) + list(legs_2), list(range(len(tensor.shape)))).reshape(one, two)

	if chi_number is not None and chi_number <= min(matrix.shape)-1:
		U, S, V = scipy.sparse.linalg.svds(matrix, k = chi_number)
	else:
		U, S, V = scipy.linalg.svd(matrix, full_matrices = False, check_finite = False)

	sort = np.argsort(S)[-chi_number:]

	U = U[:, sort]
	S = S[sort]
	V = V[sort, :]

	return U.reshape(shape_one + [chi_number]), S, V.reshape([chi_number] + shape_two)
