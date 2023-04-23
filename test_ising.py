import MainScripts as ms
import timeit

start_time = timeit.default_timer()

#ising model
calc = ms.CalcConfig()
calc.method = "trg"
calc.metModification = "default"
calc.model = "ising"
calc.lattice = "square"
calc.metParam = 10

#model params
T = 1.0 / calc.constant
chi_number = 10
for J in ms.np.arange(0.15, 0.85, 0.01):
	m_par = [0, J, 0, 0, 0, 0]
	#result = ms.full(calc, T, m_par)
	#print(J, result[0], result[1] , result[2] , result[3])
	calc_time = timeit.default_timer() - start_time
	result = ms.simulate(calc, T, m_par)
	print(J, result, calc_time)
