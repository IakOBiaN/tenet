import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = "default"
calc.model = "Pentacene_model_1_simple"
calc.lattice = "square"
calc.gen_tensor = "default"
calc.metParam = 1
#model params
T = 300.0
mu_pentacene = 0#-656.7944123
e_close = 60.7
e_one = 48.8
e_two = 38.4

print("Chemical_potential", "Coverage", "Entropy", "Susceptibility", "Heat_capacity", "Grand_potential")
coverage_res = []
for mu in ms.np.arange(-50, 200.01, 5.0):
	m_par = [mu_pentacene + mu, e_close, e_one, e_two]
	result = ms.full(calc, T, m_par, dmu = 1e-2, dT = 1e-2, T_derivative = True)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0], result[1] , result[2] , result[3], result[4])

etalon_coverage = [1.9674861973846847e-09, 1.46056853335251e-08, 1.0842522599780483e-07, 8.048874955702842e-07, 5.9746817054538265e-06, 4.4330940197772776e-05, 0.00032787774323565604, 0.0023694114930239256, 0.014754187918739059, 0.052631570010938655, 0.09161639144308897, 0.11319908943638879, 0.12440318808913942, 0.13070414750058212, 0.1345364926560011, 0.1370075883292854, 0.13866882482593823, 0.13981936041962406, 0.1406334396414355, 0.14121847655892417, 0.14164372623639565, 0.14195544498303292, 0.14218539715038503, 0.14235588321591663, 0.14248285670026561, 0.14257795637506732, 0.1426499316973917, 0.14270579620099255, 0.14275205062554974, 0.14279651591477815, 0.1428518805864698, 0.1429434439009114, 0.14312677944330332, 0.14352855417882893, 0.14444066976881942, 0.1465333909701272, 0.1513094596111486, 0.16187131139933086, 0.18284594189275083, 0.21322561785783734, 0.2368448014351543, 0.2462740934610963, 0.24891905317865337, 0.24962294458461542, 0.24983445664474857, 0.24991377885577037, 0.24995089971273643, 0.24997095063934438, 0.24998255575070516, 0.24998946603886907, 0.2499936257969182]

etalon_coverage = ms.np.array(etalon_coverage)
coverage_res = ms.np.array(coverage_res)
difference = sum(ms.np.abs(coverage_res - etalon_coverage))

assert difference < 0.5, "ERROR! Test Pentacene_model_1_simple_tm.py is broken now!"
print("Test Pentacene_model_1_simple_tm.py is", end = "")
print("\033[92m {}\033[00m" .format("OK"), "!", sep = "")
