import MainScripts as ms

#mono with interactions
method = "trg"
model = "langmuir"
lattice = "square"
#model params
T = 120.0
chi_number = 10
for mu in ms.np.arange(-10.00, 10.01, 0.5):
	m_par = [mu, 2.0, 0, 0, 0, 0]
	result = ms.full(method, model, lattice, chi_number, T, m_par)
	print(mu, result[0], result[1] , result[2] , result[3])
