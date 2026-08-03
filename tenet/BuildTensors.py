import numpy as np
from itertools import product

from tenet.models import get_model, inf

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def build_matrix (calc, temp, m_par):

	model = calc.model
	neigbours = calc.coord

	if len(m_par) < 10:
		m_par = m_par + [0.0] * (10 - len(m_par))

	models_dict = {
		"langmuir" : True,
		"langmuir_m" : True,
		"binary" : True,
		"ising" : True,
		"hard-hexagon" : True,
		"TLAT" : True,
		"dimers" : True,
		"1NN" : True,
		"2NN" : True,
		"3NN" : True,
		"4NN" : True,
		"5NN" : True,
		"qstate" : True,
		"CHD_simple" : True,
		"Pentacene_model_1_simple" : True,
		"Pentacene_model_1_complex" : True,
		"Pentacene_model_2" : True,
		"Pentacene_model_3" : True,
		"CHD_complex" : True,
		"six_leg_test" : True,
		"dimers_test" : True,
		"1D_long-range" : True,
		"2D_long-range" : True,
		"2D_long-range_V" : True
	}

	exist = models_dict.get(calc.model)
	assert (exist is not None), "Error! This model is not in the database"

	#[any],[right, bottom],[right-up, right, right-bottom], [right-up, right, right-bottom, bottom]
	matrixes = []
	build = get_model(model)
	if build is not None:
		matrixes = build(calc, temp, m_par)

	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i] / (calc.constant * temp)
		max_value = matrixes[i].max()
		first_norm = 0
		if max_value > 650:
			temp_norm = matrixes[i] / (matrixes[i] - first_norm)
			for k, a in enumerate(temp_norm):
				for m, b in enumerate(a):
					if b == 0:
						temp_norm[k][m] = 1
			matrixes[i] = np.divide(matrixes[i], temp_norm)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])

	return matrixes, first_norm
