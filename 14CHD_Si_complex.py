import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "CHD_complex"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 32
#model params
T = 300.0
mu_d_sigma = -190.0 -(-333.0) #delta of ads energies from dft calculations
mu_d_sigma = - mu_d_sigma #we operate with chemical potential in gas phase
mu_t_sigma = 0.0

e_d_d_hor_same = 24.0
e_d_d_hor_dif_in = 8.0
e_d_d_hor_dif_out = 53.0
e_d_t_hor_in = 22.0
e_d_t_hor_out = 29.0
e_t_t_hor = 43.0

e_d_d_vert_same = 14.0
e_d_d_vert_dif = 14.0
e_d_t_vert_in = 14.0
e_d_t_vert_out = 14.0
e_t_t_vert = 12.0

#for this model it is better, if we join tensors in line
calc.join_tensors = [3, 1]
print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(-20.00, 400.01, 2.0):
	m_par = [mu_t_sigma + mu, mu_d_sigma + mu, e_d_d_hor_same, e_d_d_hor_dif_in, e_d_d_hor_dif_out, e_d_t_hor_in, e_d_t_hor_out, e_t_t_hor, e_d_d_vert_same, e_d_d_vert_dif, e_d_t_vert_in, e_d_t_vert_out, e_t_t_vert]
	result = ms.full(calc, T, m_par, dmu = 1e-3, dT = 1e-3, derivatives = [1, 1], T_derivative = True)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
