import MainScripts as ms
import timeit

start_time = timeit.default_timer()

#mono with interactions
calc = ms.CalcConfig()
calc.method = "trg"
calc.model = "langmuir"
calc.lattice = "square"
calc.metParam = 10
#model params
T = 120.0
for mu in ms.np.arange(-10.00, 20.01, 0.5):
	m_par = [mu, 4.0, 0, 0, 0, 0]
	result = ms.full(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], calc_time)
