import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "2D_long-range_V"
calc.lattice = "triangular"
calc.gen_tensor = "svd"
calc.metParam = 100
#model params
T = 0.8
calc.constant = 1
sigma = 1.2
eps = 1.0

f = open("temp.dat", "w")

for mu in ms.np.arange(-8.0, 45.41, 1.0):
	m_par = [mu, [sigma, eps], 0, 0, 0, 0]
	result = ms.full(calc, T, m_par, dmu = 0.05, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	print(mu, result[0])
	print(mu, result[0], file = f, flush = True)

f.close()
