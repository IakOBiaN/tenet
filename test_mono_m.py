import MainScripts as ms
import timeit

model_name = 'TRG_IRF'

calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "langmuir_m"
calc.lattice = "triangular"
calc.gen_tensor = "IRF_m"
#calc.metParam = 16
#model params
T = 120.0

invest = [i for i in range(29, 61)]

for lat_size in invest:
	calc.metParam = lat_size
	f = open('results_' + model_name + str(lat_size) + '.dat', 'w')
	f2 = open('time_' + model_name + str(lat_size) + '.dat', 'w')
	f.write('mu coverage S sus Cp grand_potential calc_time' + '\n')

	start_time = timeit.default_timer()
	calc_time = 0

	for mu in ms.np.arange(-10.00, 60.01, 0.1):
		m_par = [mu, 6.0, 2.0, 0, 0, 0]
		result = ms.full(calc, T, m_par)
		calc_time = timeit.default_timer() - start_time
		print(mu, result[0], result[1], result[2], result[3], result[4], calc_time, lat_size)
		f.write(('{}\t' * 7).format(mu, result[0], result[1], result[2], result[3], result[4], calc_time) + '\n')
		f2.write(str(mu) + ' ' + str(calc_time) + '\n')
	f2.write("Res. calculation_time: " + str(calc_time) + '\n')
	f.close()
	f2.close()
