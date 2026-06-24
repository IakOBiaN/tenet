import Scripts.MainScripts as ms
from math import sin, pi
import timeit

def repulsions(sigma, int_distance, power = 2):
	interactions = []
	for r in range(1, int_distance + 1):
		interactions.append(sigma / (r ** power))
	return interactions

def lennard_jones(sigma, eps_lj, int_distance, power = (6, 12)):
	interactions = []
	for r in range(1, int_distance + 1):
		interactions.append(4 * eps_lj * ((sigma / r) ** power[1] - (sigma / r) ** power[0]))
	return interactions

def nonmonotonic(eps_f, q_f, delta_f, int_distance):
	interactions = []
	for r in range(1, int_distance + 1):
		interactions.append(-eps_f * (2 * sin(delta_f) / pi) ** 2 * sin(2 * q_f * r + 2 * delta_f) / (q_f * r) ** 2)
	return interactions

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "1D_long-range"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 0.1
calc.constant = 1
#interactions = repulsions(sigma = 5, int_distance = 5)
#interactions = lennard_jones(sigma = 1.782, eps_lj = 1, int_distance = 5)
#interactions = nonmonotonic(eps_f = 12.5, q_f = 1.06, delta_f = pi / 2, int_distance = 10)
interactions = [1.0, 0.57735, 0.5, 0.37796, 0.33333, 0.28868, 0.27735, 0.25, 0.22942]
for mu in ms.np.arange(-1, 10.01, 0.1):
	m_par = [mu, interactions, 0, 0, 0, 0]
	obs = ms.thermodynamics(calc, T, m_par, coverage = True)
	calc_time = timeit.default_timer() - start_time
	print(mu, obs["coverage"])
