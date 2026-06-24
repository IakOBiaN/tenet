import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

#TLAT model
calc = ms.CalcConfig()
calc.method = "htn"
calc.metModification = "default"
calc.model = "TLAT"
calc.lattice = "FSHL"
calc.metParam = 1

#model params
calc.constant = 1
T = 1.0

for param in ms.np.arange(-1.5, 0.0, 0.01):
	J1 = param
	J2 = J1
	Js = -1.0
	m_par = [0.0, J1, J2, Js, 0.0, 0.0]
	result = ms.thermodynamics(calc, T, m_par, heat_capacity = True, dT = 1e-4)["heat_capacity"]
	calc_time = timeit.default_timer() - start_time
	print(Js, result, calc_time)
