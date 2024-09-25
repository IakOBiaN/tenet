import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

calc = ms.CalcConfig()

calc.method = "tm"
calc.metModification = [2, 26]
calc.model = "dimers"
calc.lattice = "hexagonal"
calc.gen_tensor = "to_square"
calc.metParam = 3
#model params
T = 200.0
coverage_res = []
for mu in ms.np.arange(-40.00, 40.01, 2.0):
	m_par = [mu, 20.0, 0, 0, 0, 0]
	result = ms.full(calc, T, m_par, dmu = 0.05, T_derivative = False)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result[0])
	print(mu, result[0])

etalon_coverage = [8.963384563974973e-06, 2.9837243445339875e-05, 9.928015128454916e-05, 0.0003298819771264602, 0.0010910591349063141, 0.0035549498265057423, 0.011064114572148023, 0.030554871046331936, 0.06755497614911121, 0.11563123910903694, 0.16271351138537504, 0.20103059913626597, 0.22755729165348848, 0.24387527545046472, 0.25594328038095693, 0.26780641498384217, 0.2742579455947114, 0.2764884233496989, 0.2772693869498788, 0.2775673884406782, 0.277688132975884, 0.27773891681120766, 0.2777608195261472, 0.27777051491828963, 0.2777751085558844, 0.2777779593627372, 0.2777807049396941, 0.27779144963384184, 0.2778381888561654, 0.27839312427531127, 0.2934261877865474, 0.35449259077390494, 0.3612644067495019, 0.36483712029106385, 0.3994851069322536, 0.4975193589818616, 0.499993728236916, 0.4999413768309324, 0.4999999998016236, 0.4999999999405702, 0.4999999999821725]

etalon_coverage = ms.np.array(etalon_coverage)
coverage_res = ms.np.array(coverage_res)
difference = sum(ms.np.abs(coverage_res - etalon_coverage))

assert difference < 0.5, "ERROR! Test dimer_tm.py is broken now!"
print("Test dimer_tm.py is", end = "")
print("\033[92m {}\033[00m" .format("OK"), "!", sep = "")
