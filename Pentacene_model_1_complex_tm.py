import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "Pentacene_model_1_complex"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 300.0
mu_pentacene_per = -646.4
mu_pentacene_par = -607.7

mu_pentacene_per = mu_pentacene_par - mu_pentacene_per
mu_pentacene_par = 0

e_1 = 60.7
e_2 = 48.8
e_3 = 38.4
e_4 = -2.4 / 2.0
e_5 = 4.0 / 2.0
e_6 = 2.0 / 2.0
e_7 = -17.4
e_8 = 3.0
e_9 = 1.0
e_10 = -15.1
e_11 = 0
e_12 = 0


print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(-80, -0.01, 1.0):
	m_par = [mu_pentacene_per + mu, mu_pentacene_par + mu, e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10, e_11, e_12]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, derivatives = [1, 1], T_derivative = True)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
