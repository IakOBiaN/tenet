import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "btrg"
calc.metModification = -0.5
calc.model = "hard-hexagon"
calc.lattice = "triangular"
calc.gen_tensor = "default"
calc.metParam = 24
calc.constant = 1
#model params
T = 1.0
coverage_res = []
for mu in ms.np.arange(-6.00, 6.01, 0.1):
	m_par = [mu, 0.0, 0, 0, 0, 0]
	obs = ms.thermodynamics(calc, T, m_par, coverage = True, susceptibility = True, entropy = True, heat_capacity = True)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(obs["coverage"])
	print(mu, obs["coverage"], obs["grand_potential"])
