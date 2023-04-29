import MainScripts as ms
import timeit

start_time = timeit.default_timer()

#ising model
calc = ms.CalcConfig()
calc.method = "hotrg"
calc.metModification = "default"
calc.model = "ising"
calc.lattice = "square"
calc.metParam = 10

#model params
calc.constant = 1
for T in ms.np.arange(2.1, 2.501, 0.01):
	m_par = [0, 1, 0, 0, 0, 0]
	result = ms.simulate(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(T, result, calc_time)
