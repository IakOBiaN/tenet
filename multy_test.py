import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "six_leg_test"
calc.lattice = "triangular"
calc.gen_tensor = "six_leg_tensor"
calc.metParam = 16
#model params
T = 300
mu_Cu = -10.0
w2 = -10
w3 = -18
w3_1 = -16
w4 = -19
for mu_TPB in ms.np.arange(30.0, 31.1, 2.0):
	m_par = [mu_TPB, mu_Cu, w2, w3, w3_1, w4]
	result = ms.full(calc, T, m_par, derivatives = [1, 1], T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu_TPB, result[0])
