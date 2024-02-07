import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "Pentacene_model_1"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 300.0
mu_pentacene = 0#-656.7944123
inf = 1e10
e_close = 60.7
e_one = 48.8
e_two = 38.4

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(-50, 200.01, 5.0):
	m_par = [mu_pentacene + mu, e_close, e_one, e_two]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, T_derivative = True)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
