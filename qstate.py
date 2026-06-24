import Scripts.MainScripts as ms
import timeit

start_time = timeit.default_timer()

#qstate_model
calc = ms.CalcConfig()

calc.method = "trg"
calc.metModification = "default"
calc.model = "qstate"
calc.lattice = "square"
calc.metParam = 10

#model params
calc.constant = 1
c = 1
n = 12
epsilon = -1
delta = 1.0
T = 0.05
dmu = 0.01
dT = 0.01
filename = calc.model + "_c=" + str(c) + "_n=" + str(n) + "_d=" + str(delta) + "_T=" + str(T) + "_chi=" + str(calc.metParam) + "_dm=" + str(dmu) + "_dt=" + str(dT) + ".dat"
file = open(filename, "w")
print("chemical_potential", "entropy", "susceptibility", "heat_capacity", "free_energy", file = file)
for mu in ms.np.arange(-1.00, 7.01, 0.2):
	m_par = [mu, c, n, epsilon, delta, 0.0]
	obs = ms.thermodynamics(calc, T, m_par, coverage = True, susceptibility = True, entropy = True, heat_capacity = True)
	print(mu, obs["coverage"], obs["entropy"], obs["susceptibility"], obs["heat_capacity"], obs["grand_potential"])
	#print(mu, obs["coverage"], obs["entropy"], obs["susceptibility"], obs["heat_capacity"], obs["grand_potential"], file = file)
file.close()
