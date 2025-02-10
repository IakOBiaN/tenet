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
	result = ms.full(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0], result[4])
