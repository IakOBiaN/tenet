import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "six_leg_test"
calc.lattice = "triangular"
calc.gen_tensor = "six_leg_tensor"
calc.metParam = 48
#model params
T = 300
for mu in ms.np.arange(-2.0, 20.1, 2.0):
	m_par = [mu, -10.0, -10.0, 0, 0, 0]
	result = ms.full(calc, T, m_par, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0])
