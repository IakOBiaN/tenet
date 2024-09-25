import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "Pentacene_model_3"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 100
#model params
T = 1000.0

mu_pentacene_per = -651.2
mu_pentacene_par = -583.0

mu_pentacene_per = mu_pentacene_par - mu_pentacene_per
mu_pentacene_par = 0

e_v1 = -10.5 # -12.0
e_v2 = -52.9 # -56.6
e_v3 = -45.9 # -52.1
e_v4 = -2.3
e_v5 = 28.7
e_v6 = -49.5
e_v7 = 97.2
e_v8 = -26.2 # -28.5
e_v9 = 14.0

e_h1 = 7.9 # 8.5
e_h2 = -7.1 # -1.8
e_h3 = 11.6
e_h4 = 20.1

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
coverage_res = []
for mu in ms.np.arange(-100, 200.01, 2.5):
	m_par = [mu_pentacene_per + mu, mu_pentacene_par + mu, e_v1, e_v2, e_v3, e_v4, e_v5, e_v6, e_v7, e_h1, e_h2, e_h3, e_h4, e_v8, e_v9]
	result = ms.full(calc, T, m_par, dmu = 1e-1, dT = 1e-2, derivatives = [1, 1], T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0], result[1] , result[2] , result[3], result[4])

print(coverage_res)
