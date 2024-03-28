import Scripts.MainScripts as ms
from scipy.optimize import minimize

#ising model
calc = ms.CalcConfig()
calc.method = "htn"
calc.metModification = "default"
calc.model = "binary"
calc.lattice = "FSHL"
calc.metParam = 1


#model params
f = open("data2.dat", "w")
T = 100.0
muA = -10
muB = -10
maximum = []
for muB in ms.np.arange(-5.00, 5.01, 0.1):
	max_counter = 0
	old_value = -1000
	old_A = 0
	old_B = 0
	for muA in ms.np.arange(10.00, 20.01, 0.01):
		m_par = [muA, muB, 4.0, 6.0, 0, 0]
		result = ms.susceptibility(calc, T, m_par)
		if result > old_value:
			old_value = result
			old_A = muA
			old_B = muB
			max_counter = 0
		else:
			max_counter += 1
		if max_counter == 5:
			maximum.append([old_B, old_A, old_value])
			print(old_B, old_A, old_value, file = f, flush = True)
		print(muA, muB, result)

for aaa in maximum:
	for bbb in aaa:
		print(bbb, end = " ")
	print()

f.close()
