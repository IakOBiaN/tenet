import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = [2, 26]
calc.model = "dimers"
calc.lattice = "hexagonal"
calc.gen_tensor = "to_square"
calc.metParam = 3
#model params
T = 200.0
for mu in ms.np.arange(-40.00, 40.01, 2.0):
	m_par = [mu, 20.0, 0, 0, 0, 0]
	result = ms.full(calc, T, m_par, dmu = 0.05, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0])
