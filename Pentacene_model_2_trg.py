import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "Pentacene_model_2"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 300.0
inf = -1e8
mu_2 = -(-526.5 -(-656.79))
mu_3 = -(-601.73-(-656.79))
mu_4 = 0



e_44_1 = 60.7
e_44_2 = 48.8
e_44_3 = 38.4

e_22_1 = 199.5
e_22_2 = 217.4
e_22_3 = 197.8
e_22_4 = 101.7
e_22_5 = 90.3
e_22_6 = 150.5
e_22_7 = 15.0
e_22_8 = 97.9
e_22_9 = 82.1

e_33_1 = 206.3
e_33_2 = 295.6
e_33_3 = 236.0
e_33_4 = 95.7
e_33_5 = 84.6
e_33_6 = 98.1
e_33_7 = 87.3
e_33_8 = 109.3
e_33_9 = 82.0

e_24_1 = 106.3
e_24_2 = 87.9
e_24_3 = 59.9
e_24_4 = 62.7
e_24_5 = 15.0
e_24_6 = 68.9

e_34_1 = 164.9
e_34_2 = 112.8
e_34_3 = 75.1
e_34_4 = 62.8
e_34_5 = 69.4
e_34_6 = 60.6

e_23_1 = 180.2
e_23_2 = 182.4
e_23_3 = 205.3
e_23_4 = 201.3
e_23_5 = 91.7
e_23_6 = 96.5
e_23_7 = 89.5
e_23_8 = 94.9
e_23_9 = 89.2
e_23_10 = 81.8
e_23_11 = 91.3
e_23_12 = 77.3

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
for mu in ms.np.arange(-50, 1000.01, 5.0):
	m_par = [mu_2 + mu, mu_3 + mu, mu_4 + mu, e_44_1, e_44_2, e_44_3, \
		e_22_1, e_22_2, e_22_3, e_22_4, e_22_5, e_22_6, e_22_7, e_22_8, e_22_9, \
		e_33_1, e_33_2, e_33_3, e_33_4, e_33_5, e_33_6, e_33_7, e_33_8, e_33_9, \
		e_24_1, e_24_2, e_24_3, e_24_4, e_24_5, e_24_6, \
		e_34_1, e_34_2, e_34_3, e_34_4, e_34_5, e_34_6, \
		e_23_1, e_23_2, e_23_3, e_23_4, e_23_5, e_23_6, e_23_7, e_23_8, e_23_9, e_23_10, e_23_11, e_23_12]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, derivatives = [1, 1, 1], T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0], result[1] , result[2] , result[3], result[4])
