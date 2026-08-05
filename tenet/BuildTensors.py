import numpy as np

from tenet.models import get_model

def build_matrix (calc, temp, m_par):
	"""Boltzmann weight matrices for calc.model, ready for the tensor network.

	Looks the model up in the registry (see tenet/models/) and applies the
	epilogue every model shares: divide the energies by R*T and exponentiate.

	Returns (matrixes, first_norm).  first_norm is always 0 - the loop resets it
	on every pass and nothing ever assigns anything else - but it is still
	returned because simulate() seeds its running scale with it.
	"""
	build = get_model(calc.model)
	assert (build is not None), "Error! This model is not in the database"

	if len(m_par) < 10:
		m_par = m_par + [0.0] * (10 - len(m_par))

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
