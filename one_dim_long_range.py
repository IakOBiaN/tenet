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
T = 120.0
coverage_res = []
for mu in ms.np.arange(-5.0, 20.01, 0.5):
	m_par = [mu, [4.0, 2.0, 1.0], 0, 0, 0, 0, 0, 0, 0, 3]
	result = ms.full(calc, T, m_par, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0])
