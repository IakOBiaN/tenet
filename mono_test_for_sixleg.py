import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "langmuir_test"
calc.lattice = "triangular"
calc.gen_tensor = "six_leg_tensor"
calc.metParam = 16
#model params
T = 200.0
for mu in ms.np.arange(-8.00, 23.01, 2.0):
	m_par = [mu, 4.0, 0, 0, 0, 0]
	result = ms.full(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], calc_time)
