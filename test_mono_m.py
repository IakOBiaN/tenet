import MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "langmuir_m"
calc.lattice = "triangular"
calc.gen_tensor = "default_m"
calc.metParam = 16
#model params
T = 120.0
for mu in ms.np.arange(-8.00, 23.01, 0.5):
	m_par = [mu, 4.0, 0, 0, 0, 0]
	result = ms.full(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], calc_time)
