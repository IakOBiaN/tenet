"""Spin and orientational models, including the standard benchmarks."""
from math import pi, cos, radians

import numpy as np

from tenet.models import register


@register("ising")
def ising(calc, temp, m_par):
	"""Two-dimensional Ising model.

	m_par: 0 - external field, 1 - coupling
	"""
	neigbours = calc.coord
	matrixes = [np.array([[(m_par[1] - m_par[0] / (neigbours / 2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1] + m_par[0] / (neigbours / 2.0))]]), ] * 3
	return matrixes


@register("TLAT")
def tlat(calc, temp, m_par):
	"""TLAT: four-state model with three interaction constants.

	m_par: 1, 2, 3 - the interaction constants.  m_par[0] is not read by this
	construction.
	"""
	matrixes = [np.array([[-m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3], m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3]], [-m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3], m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3]], [m_par[1] - m_par[2] + m_par[3], m_par[1] + m_par[2] - m_par[3], -m_par[1] - m_par[2] - m_par[3], -m_par[1] + m_par[2] + m_par[3]], [m_par[1] + m_par[2] - m_par[3], m_par[1] - m_par[2] + m_par[3], -m_par[1] + m_par[2] + m_par[3], -m_par[1] - m_par[2] - m_par[3]]]), ] * 3
	return matrixes


@register("qstate")
def qstate(calc, temp, m_par):
	"""Orientational model with n discrete orientations per site.

	m_par: 0 - mu, 1 - c (arms per particle), 2 - n (orientations),
	       3 - epsilon (arm-arm interaction), 4 - delta (isotropic offset)

	Builds the three direction-dependent matrices (right-up, right,
	right-bottom).  Note the chemical potential is divided by a literal 6 here
	rather than by ``calc.coord``.
	"""
	mu = m_par[0] / 6
	c = m_par[1]
	n = m_par[2]
	epsilon = m_par[3]
	delta = m_par[4]
	matrixes = []
	#right-up
	anglesi = [(i - radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	anglesj = [(i + pi - radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	matrix = [[0, ] + [mu, ] * n]
	for alpha_i in anglesi:
		line = [mu, ]
		for alpha_j in anglesj:
			uij = 0
			for k in range(c):
				for l in range (c):
					if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
						uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
			uij += delta
			line.append(-uij + 2.0 * mu)
		matrix.append(line)
	matrixes.append(np.array(matrix))
	#right
	anglesi = [i for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	anglesj = [(i - pi) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	matrix = [[0, ] + [mu, ] * n]
	for alpha_i in anglesi:
		line = [mu, ]
		for alpha_j in anglesj:
			uij = 0
			for k in range(c):
				for l in range (c):
					if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
						uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
			uij += delta
			line.append(-uij + 2.0 * mu)
		matrix.append(line)
	matrixes.append(np.array(matrix))
	#right-bottom
	anglesi = [(i + radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	anglesj = [(i + pi + radians(60)) for i in np.arange(0, 2.0 * pi, 2.0 * pi / n)]
	matrix = [[0, ] + [mu, ] * n]
	for alpha_i in anglesi:
		line = [mu, ]
		for alpha_j in anglesj:
			uij = 0
			for k in range(c):
				for l in range (c):
					if (cos(alpha_i - 2 * pi * k / c) > 0) and (cos(alpha_j - 2 * pi * l / c) > 0):
						uij += epsilon * cos(alpha_i - 2 * pi * k / c) ** 2 * cos(alpha_j - 2 * pi * l / c) ** 2
			uij += delta
			line.append(-uij + 2.0 * mu)
		matrix.append(line)
	matrixes.append(np.array(matrix))
	return matrixes
