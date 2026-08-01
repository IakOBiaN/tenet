"""Hard-core exclusion series: 1NN through 5NN."""
import numpy as np

from tenet.models import inf, register


@register("1NN")
@register("2NN")
@register("3NN")
@register("4NN")
@register("5NN")
def exclusion_series(calc, temp, m_par):
	"""Lattice gas that forbids every neighbour shell closer than the k-th.

	m_par: 0 - mu.  The registered name supplies k: "1NN" excludes nothing beyond
	double occupancy, "5NN" forbids the first four shells.

	All five models share one set of three 8x8 matrices; they differ only in which
	entries are switched from an allowed weight to the ``inf`` sentinel.  Shell j is
	forbidden exactly when j < k, which the original spelled out as a 41-line
	if/elif over the five names.
	"""
	neigbours = calc.coord
	k = int(calc.model[0])

	var_1NN_0 = inf if k > 1 else 0
	var_1NN_mu = inf if k > 1 else m_par[0] / neigbours
	var_2NN_0 = inf if k > 2 else 0
	var_2NN_mu = inf if k > 2 else m_par[0] / neigbours
	var_3NN = inf if k > 3 else 0
	var_4NN = inf if k > 4 else 0

	matrixes = [np.array([[0, inf, 0, inf, inf, inf, 0, 0], \
					[inf, inf, var_1NN_mu, inf, m_par[0] / neigbours, inf, var_1NN_mu, var_2NN_mu], \
					[inf, inf, inf, 0, inf, var_1NN_0, var_2NN_0, var_1NN_0], \
					[0, var_1NN_mu, var_2NN_0, inf, inf, var_2NN_0, var_3NN, var_3NN], \
					[0, var_2NN_mu, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN, var_4NN], \
					[0, var_1NN_mu, var_3NN, var_2NN_0, inf, inf, var_2NN_0, var_3NN], \
					[inf, inf, var_2NN_0, var_1NN_0, inf, 0, inf, var_1NN_0], \
					[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf]]), \
				np.array([[0, inf, 0, 0, inf, inf, inf, 0], \
					[inf, inf, var_2NN_mu, var_1NN_mu, inf, m_par[0] / neigbours, inf, var_1NN_mu], \
					[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf], \
					[inf, inf, var_1NN_0, inf, 0, inf, var_1NN_0, var_2NN_0], \
					[0, var_1NN_mu, var_3NN, var_2NN_0, inf, var_1NN_0, var_2NN_0, var_3NN], \
					[0, var_2NN_mu, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0, var_3NN], \
					[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
					[inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0, inf]]), \
				np.array([[0, inf, 0, 0, 0, inf, inf, inf], \
					[inf, inf, var_1NN_mu, var_2NN_mu, var_1NN_mu, inf, m_par[0] / neigbours, inf], \
					[inf, inf, inf, var_1NN_0, var_2NN_0, var_1NN_0, inf, 0], \
					[inf, m_par[0] / neigbours, inf, inf, inf, inf, inf, inf], \
					[inf, inf, var_2NN_0, var_1NN_0, inf, 0, inf, var_1NN_0], \
					[0, var_1NN_mu, var_3NN, var_3NN, var_2NN_0, inf, inf, var_2NN_0], \
					[0, var_2NN_mu, var_3NN, var_4NN, var_3NN, var_1NN_0, inf, var_1NN_0], \
					[0, var_1NN_mu, var_2NN_0, var_3NN, var_3NN, var_2NN_0, inf, inf]])]
	return matrixes
