import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

#binary gas
calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "binary"
calc.lattice = "square"
calc.metParam = 10
#model params
T = 100.0
for mu in ms.np.arange(-40.00, 30.01, 1.0):
	m_par = [mu, 10.0, 4.0, 6.0, 0, 0]
	result = ms.full(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], calc_time)
