import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "long-range"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 0.1
calc.constant = 1
coverage_res = []
for mu in ms.np.arange(-1, 10.01, 0.1):
	m_par = [mu, [1.0, 0.57735, 0.5, 0.37796, 0.33333, 0.28868, 0.27735, 0.25, 0.22942, 0.21822], 0, 0, 0, 0]
	result = ms.full(calc, T, m_par, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0])
