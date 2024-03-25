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
T = 100.0
muB = -10
for muA in ms.np.arange(-10.00, 25.01, 0.2):
	m_par = [muA, muB, 4.0, 6.0, 0, 0]
	result = minimize(lambda x: -ms.susceptibility(calc, T, [x[0]] + m_par[1:], derivatives = [1, 1]), 0, tol = 0.001)
	print(muA, result.x[0])
