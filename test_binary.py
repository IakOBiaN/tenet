import MainScripts as ms

#binary gas
method = "trg"
model = "binary"
lattice = "square"
#model params
T = 100.0
chi_number = 10
for mu in ms.np.arange(-40.00, 30.01, 1.0):
	m_par = [mu, 10.0, 4.0, 6.0, 0, 0]
	result = ms.full(method, model, lattice, chi_number, T, m_par)
	print(mu, result[0], result[1] , result[2] , result[3])
