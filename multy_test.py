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
T = 640#39.69208563868
eps = 4.8
mu_Cu = -19.0
w2 = -20
w3 = -40.0
w3_1 = -35.2 - eps
w4 = -45.6 - 3.0 * eps
mu_TPB = 6.0

"""eps = 1.7
mu_Cu = -19.0
w2 = -20
w3 = -35.4
w3_1 = -31.5 - eps
w4 = -37.7 - 3.0 * eps
mu_TPB = 6.0"""

fff = open("an_calc_prov.dat", "w")
for T in ms.np.arange(300, 10000.01, 100):
	m_par = [mu_TPB, mu_Cu, w2, w3, w3_1, w4, eps]
	result = ms.full(calc, T, m_par, dmu = 1e-1, derivatives = [1, 1], T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(T, result[0])
	print(mu_TPB, result[0], file = fff, flush = True)
fff.close()
