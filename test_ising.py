import MainScripts as ms

#ising model
method = "trg"
model = "ising"
lattice = "square"
#model params
T = 1.0 / constant
chi_number = 10
for J in np.arange(0.15, 0.85, 0.01):
	m_par = [0, J, 0, 0, 0, 0]
	#result = full(method, model, lattice, chi_number, T, m_par)
	#print(J, result[0], result[1] , result[2] , result[3])
	result = ms.calc(method, model, lattice, T, m_par, chi_number)
	print(J, result)
