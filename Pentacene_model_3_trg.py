import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "Pentacene_model_3"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 32
#model params
T = 300.0
mu_pentacene_par = -607.7
mu_pentacene_per = -646.4

mu_pentacene_par = -(mu_pentacene_par - mu_pentacene_per)

mu_pentacene_per = 0

e_v1 = 31.9
e_v2 = 19.8
e_v3 = 5.0
e_v4 = 40.1
e_v5 = 101.4
e_v6 = 6.0
e_v7 = 60.7

e_h1 = -2.4
e_h2 = -17.4
e_h3 = -15.1
e_h4 = 60.7

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(-100, 400.01, 5.0):
	m_par = [mu_pentacene_per + mu, mu_pentacene_par + mu, e_v1, e_v2, e_v3, e_v4, e_v5, e_v6, e_v7, e_h1, e_h2, e_h3, e_h4]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, derivatives = [1, 1], T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
