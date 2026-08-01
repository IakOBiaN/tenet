"""Lattice gas models: site occupation, with and without extra structure."""
import numpy as np

from tenet.models import inf, register


@register("langmuir")
def langmuir(calc, temp, m_par):
	"""Langmuir lattice gas: one particle per site, nearest-neighbour interaction.

	m_par: 0 - mu, 1 - eps
	"""
	neigbours = calc.coord
	matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3
	return matrixes


@register("langmuir_m")
def langmuir_m(calc, temp, m_par):
	"""Langmuir gas with an added three-particle interaction.

	m_par: 0 - mu, 1 - eps, 2 - multiparticle interaction

	Returns a fourth, rank-3 weight; the ``default_m`` and ``IRF_m`` tensor
	construction variants pick it up as ``matrixes[-1]``.
	"""
	neigbours = calc.coord
	mult = np.zeros((2, 2, 2))
	mult[1, 1, 1] = -m_par[2]
	matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3 + [mult]
	return matrixes


@register("binary")
def binary(calc, temp, m_par):
	"""Two-component (A/B) lattice gas.

	m_par: 0 - muA, 1 - muB, 2 - epsAA, 3 - epsBB

	The original comment also listed ``4 - epsAB``, but the construction never
	reads it: an A-B pair carries the two chemical potentials and no interaction
	energy.  Left as is - adding the term would change the physics.
	"""
	neigbours = calc.coord
	matrixes = [np.array([[0.0, m_par[0] / neigbours, m_par[1] / neigbours], [m_par[0] / neigbours, -m_par[2] + 2.0 * m_par[0] / neigbours, (m_par[0] + m_par[1]) / neigbours], [m_par[1] / neigbours, (m_par[0] + m_par[1]) / neigbours, -m_par[3] + 2.0 * m_par[1] / neigbours]]) ,] * 3
	return matrixes


@register("hard-hexagon")
def hard_hexagon(calc, temp, m_par):
	"""Hard hexagon model: neighbouring sites cannot both be occupied.

	m_par: 0 - mu.  The forbidden entry uses the ``inf`` sentinel.
	"""
	neigbours = calc.coord
	matrixes = [np.array([[0.0, m_par[0] / (neigbours)],[m_par[0] / (neigbours), inf + m_par[0]]]), ] * 3
	return matrixes


@register("dimers")
def dimers(calc, temp, m_par):
	"""Dimers on a honeycomb lattice, five states per site.

	m_par: 0 - mu, 1 - the energy gained when two sites form a dimer.

	Pins ``neigbours`` to 3 (the honeycomb coordination number) whatever
	``calc.coord`` says, and reports two merged sites via ``calc.nodes``.
	"""
	neigbours = calc.coord
	neigbours = 3.0
	calc.nodes = 2.0
	matrixes = [np.array([[0, inf, (m_par[0] + m_par[1]) / (neigbours * 2.0), (m_par[0] + m_par[1]) / (neigbours * 2.0), m_par[0] / neigbours], \
					[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
					[inf, (m_par[0] + m_par[1]) / neigbours, inf, inf, inf], \
					[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
					[m_par[0] / neigbours, inf, inf, inf, inf]]), \
				np.array([[0, (m_par[0] + m_par[1]) / (neigbours * 2.0), (m_par[0] + m_par[1]) / (neigbours * 2.0), inf, m_par[0] / neigbours], \
					[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
					[(m_par[0] + m_par[1]) / (neigbours * 2.0), inf, inf, inf, inf], \
					[inf, inf, inf, (m_par[0] + m_par[1]) / neigbours, inf], \
					[m_par[0] / neigbours, inf, inf, inf, inf]])]
	return matrixes


@register("dimers_test")
def dimers_test(calc, temp, m_par):
	"""Six-state dimer variant kept as a cross-check of the dimers construction.

	m_par: 0 - mu.  No driver script; covered by the build_matrix golden.
	"""
	neigbours = calc.coord
	mu = m_par[0] / neigbours
	matrixes = [np.array([[0, mu / 2.0, inf, mu / 2.0, mu / 2.0, mu], \
					[inf, inf, mu, inf, inf, inf], \
					[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
					[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
					[mu / 2.0, mu, inf, mu, mu, 3.0 / 2.0 * mu], \
					[mu, 3.0 / 2.0 * mu, inf, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, 2.0 * mu]]), \
				np.array([[0, mu / 2.0, mu / 2.0, mu / 2.0, inf, mu], \
					[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
					[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
					[inf, inf, inf, inf, mu, inf], \
					[mu / 2.0, mu, mu, mu, inf, 3.0 / 2.0 * mu], \
					[mu, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, 3.0 / 2.0 * mu, inf, 2.0 * mu]])]
	return matrixes
