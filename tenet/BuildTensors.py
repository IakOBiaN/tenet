import numpy as np

from tenet.models import get_model, make_params

def build_matrix (calc, temp, m_par):
	"""Boltzmann weight matrices for calc.model, ready for the tensor network.

	Looks the model up in the registry (see tenet/models/) and applies the
	epilogue every model shares: divide the energies by R*T and exponentiate.

	``m_par`` may be a positional list, as the entry scripts pass it, or a dict
	keyed by the parameter names the model declared; either is resolved against
	the model's declaration before the builder sees it.

	Returns (matrixes, first_norm).  first_norm is always 0 - the loop resets it
	on every pass and nothing ever assigns anything else - but it is still
	returned because simulate() seeds its running scale with it.
	"""
	build = get_model(calc.model)
	assert (build is not None), "Error! This model is not in the database"

	matrixes = build(calc, temp, make_params(calc.model, m_par))

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
