import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

#ising model
calc = ms.CalcConfig()
calc.method = "hotrg"
calc.metModification = "default"
calc.model = "ising"
calc.lattice = "square"
calc.metParam = 10

#model params
calc.constant = 1
coverage_res = []
for T in ms.np.arange(2.1, 2.501, 0.01):
	m_par = [0, 1, 0, 0, 0, 0]
	result = ms.simulate(calc, T, m_par)
	calc_time = timeit.default_timer() - start_time
	coverage_res.append(result)
	print(T, result, calc_time)

etalon_coverage = [2.068841298148887, 2.0708013015173705, 2.0728083987972252, 2.0748639232844015, 2.0769693006190137, 2.079126062975112, 2.0813358667451807, 2.083600514939215, 2.085921986094176, 2.088302472456617, 2.090744431912025, 2.0932506613439594, 2.095824405511297, 2.0984695289926933, 2.101190810115959, 2.103994509955232, 2.106889761103934, 2.109895185775115, 2.113038764309302, 2.1162870694923273, 2.119623799936836, 2.123040028503348, 2.1265291389652603, 2.130086303389422, 2.1337075323717767, 2.1373893779481734, 2.1411286298370302, 2.1449228208168316, 2.1487696110003034, 2.1526668862369975, 2.156612720249088, 2.160605345817893, 2.164643198249558, 2.168724630261383, 2.172848301440499, 2.1770128962034887, 2.18121723339908, 2.1854600380074953, 2.1897402656619045, 2.194057114595937, 2.198409116687249]

etalon_coverage = ms.np.array(etalon_coverage)
coverage_res = ms.np.array(coverage_res)
difference = sum(ms.np.abs(coverage_res - etalon_coverage))

assert difference < 0.5, "ERROR! Test ising.py is broken now!"
print("Test ising.py is", end = "")
print("\033[92m {}\033[00m" .format("OK"), "!", sep = "")
