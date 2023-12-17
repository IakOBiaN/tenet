import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

#ising model
calc = ms.CalcConfig()
calc.method = "htn"
calc.metModification = "default"
calc.model = "ising"
calc.lattice = "FSHL"
calc.metParam = 10

#model params
calc.constant = 1
T = 1.0
for J in ms.np.arange(0.15, 0.85, 0.01):
	m_par = [0, J, 0, 0, 0, 0]
	result = ms.heat_capasity(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	print(J, result, calc_time)
