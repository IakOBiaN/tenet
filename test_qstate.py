import MainScripts as ms

#qstate_model
method = "trg"
model = "qstate"
lattice = "triangular"
#model params
c = 1
n = 12
epsilon = -1
delta = 1.0
T = 0.05
chi_number = 10
dmu = 0.01
dT = 0.01
filename = model + "_c=" + str(c) + "_n=" + str(n) + "_d=" + str(delta) + "_T=" + str(T) + "_chi=" + str(chi_number) + "_dm=" + str(dmu) + "_dt=" + str(dT) + ".dat"
file = open(filename, "w")
print("chemical_potential", "entropy", "susceptibility", "heat_capacity", "free_energy", file = file)
for mu in np.arange(-1.00, 7.01, 0.2):
	m_par = [mu, c, n, epsilon, delta, 0.0]
	coef = 2.0/1.00
	result = full(method, model, lattice, chi_number, T, m_par)
	print(mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3], coef*result[4])
	print(mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3], coef*result[4], file = file)
file.close()
