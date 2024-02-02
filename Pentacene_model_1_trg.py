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
mu_pentacene = -656.7944123
inf = 1e10
e_close = inf#60.7
e_one = 0#48.8
e_two = 0#38.4

#for this model it is better, if we join tensors in line
#calc.join_tensors = [2, 1]

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(600.00, 800.01, 4.0):
	m_par = [mu_pentacene + mu, e_close, e_one, e_two]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
