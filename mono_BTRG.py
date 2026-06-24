import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "btrg"
calc.metModification = "default"
calc.model = "langmuir"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 16
#model params
T = 120.0
coverage_res = []
for mu in ms.np.arange(-8.00, 23.01, 0.5):
	m_par = [mu, 4.0, 0, 0, 0, 0]
	obs = ms.thermodynamics(calc, T, m_par, coverage = True, susceptibility = True, entropy = True, heat_capacity = True)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(obs["coverage"])
	print(mu, obs["coverage"], obs["entropy"], obs["susceptibility"], obs["heat_capacity"], calc_time)
