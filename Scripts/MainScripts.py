from math import exp, log, sqrt
import numpy as np
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import Scripts.TensorNetworks as tn
import Scripts.BuildTensors as bt

class CalcConfig:

	def __init__(self, methodTolerance = 1e-8, constant = 0.008314, method = "trg", metModification = "default", scale = 4, iterations = 300, model = "ising", lattice = "square", gen_tensor = "default", nodes = 1 , coord = 4, metParam = 10, join_tensors = [1, 1]):
		#tolerance of method
		self.methodTolerance = methodTolerance
		#constant. Default is R = 0.008314
		self.constant = constant
		#method for partition function calculation
		self.method = method
		#internal method modification
		self.metModification = metModification
		#scale of the method iteration. For trg and square lattice it is 2 * 2
		self.scale = scale
		#number of method iterations
		self.iterations = iterations
		#model for tensor network constructions
		self.model = model
		#lattice geometry
		self.lattice = lattice
		#method of tensor network generation
		self.gen_tensor = gen_tensor
		#number of initial nodes for first tensor
		self.nodes = nodes
		#coordination number of a lattice
		self.coord = coord
		#method parameter. For trg it is chi
		self.metParam = metParam
		#joining nodes
		self.join_tensors = join_tensors

def simulate(calc, T = 1.0, m_par = [0.0] * 10):

	if calc.lattice == "triangular":
		calc.coord = 6
	if calc.metParam == 1 and calc.method == "tm":
		calc.coord = 2

	matrixes, first_norm = bt.build_matrix(calc, T, m_par)
	if calc.method != "htn" and not (calc.method == "tm" and calc.metParam == 1):
		tensors = tn.build_tensor(calc, matrixes)

	scale = first_norm
	old_scale = -1.0
	norm = 0

	minimum_iterations = 5

	covergence = [-1e8, ]
	for i in range(calc.iterations):
		if calc.method == "trg":
			(tensors, scale, norm) = tn.trg_step(tensors, scale, norm, calc)
		elif calc.method == "btrg":
			(tensors, scale, norm) = tn.btrg_step(tensors, scale, norm, calc)
		elif calc.method == "hotrg":
			(tensors, scale, norm) = tn.hotrg_step(tensors, scale, norm, calc)
		elif calc.method == "htn":
			calc.scale = i
			(tensors, scale, norm) = tn.htn_step(matrixes, scale, norm, calc)
			covergence.append((scale + log(norm)) / (calc.nodes / (calc.constant * T)))
			if abs(covergence[-2] - covergence[-1]) < (calc.methodTolerance / 100) and i > minimum_iterations:
				break
		elif calc.method == "tm":
			calc.scale = 2
			if calc.metParam == 1:
				(tensors, scale, norm) = tn.tm_step(matrixes, scale, norm, calc)
			else:
				(tensors, scale, norm) = tn.tm_step(tensors, scale, norm, calc)
		else:
			assert False, "Error! There is no such method."

		if calc.method != "htn":
			if (abs(old_scale - scale / calc.scale) < calc.methodTolerance and i > minimum_iterations) or (tensors[0].size < 2):
				break
			else:
				old_scale = scale
	if i > 250:
		print("Warning! More than 250 iterations")
	nodes = calc.nodes * calc.join_tensors[0] * calc.join_tensors[1]
	if calc.method == "tm":
		nodes *= calc.metParam
		if calc.metModification != "default":
			nodes *= 2 ** (calc.metModification[0] - 1)
	if calc.method != "htn":
		nodes *= calc.scale ** (i + 1)
	if norm <= 0:
		norm = 1
	return (scale + log(norm)) / (nodes / (calc.constant * T))

def heat_capasity(calc, T = 1., m_par = [0.0] * 10, dT = 1e-4):
	#central second-order finite difference of the grand potential with respect to T
	#reproduces scipy.misc.derivative(..., n = 2, dx = 1e-4), removed in SciPy 1.12
	omega_minus = simulate(calc, T - dT, m_par)
	omega_center = simulate(calc, T, m_par)
	omega_plus = simulate(calc, T + dT, m_par)
	result = T * (omega_minus - 2.0 * omega_center + omega_plus) / dT ** 2
	return result

def susceptibility(calc, T = 1., m_par = [0.0]*10, dmu = 1e-4, derivatives = [1, ] + [0] * 2):
	grandPotential_dmu = []
	der_par = m_par[:]
	for i, par in enumerate(derivatives):
		if par == 1:
			der_par[i] -= dmu
	for _ in range(3):
		lnZ = simulate(calc, T, der_par)
		grandPotential_dmu.append(lnZ)
		for i, par in enumerate(derivatives):
			if par == 1:
				der_par[i] += dmu
	del der_par
	result = calc.constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dmu[1] + grandPotential_dmu[2]) / (dmu ** 2.0)
	return result

def coverage(method, model, lattice, temp = 1., m_par = [0.0]*10, temp_size = 300):
	BTP = []
	mu_step = 1e-3
	for mu_TRG in [m_par[0] - mu_step, m_par[0] + mu_step]:
		lnZ = simulate(method, model, lattice, temp, [mu_TRG] + m_par[1:])
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(mu_step*2.0)
	return result

def entropy(method, model, lattice, temp = 1., m_par = [0.0]*10):
	BTP = []
	temp_step = 1e-3
	for temp_TRG in [temp - temp_step, temp + temp_step]:
		lnZ = simulate(method, model, lattice, temp_TRG, m_par)
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(temp_step*2.0)
	return result

def full(calc, T = 1., m_par = [0.0] * 10, dmu = 1e-3, dT = 1e-3, derivatives = [1, ] + [0] * 2, T_derivative = True, mu_derivative = True):
	grandPotential_dmu = []
	grandPotential_dT = []
	if mu_derivative:
		der_par = m_par[:]
		for i, par in enumerate(derivatives):
			if par == 1:
				der_par[i] -= dmu
		for _ in range(2):
			lnZ = simulate(calc, T, der_par)
			grandPotential_dmu.append(lnZ)
			for i, par in enumerate(derivatives):
				if par == 1:
					der_par[i] += 2.0 * dmu
		del der_par
	else:
		grandPotential_dmu = [0, 0]
	if T_derivative:
		for diff_T in [T - dT, T, T + dT]:
			lnZ = simulate(calc, diff_T, m_par)
			grandPotential_dT.append(lnZ)
	else:
		grandPotential_dT = [0, 0, 0]

	coverage = - (grandPotential_dmu[0] - grandPotential_dmu[1]) / (dmu * 2.0)
	entropy = - (grandPotential_dT[0] - grandPotential_dT[2]) / (dT * 2.0)
	susceptibility = calc.constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dT[1] + grandPotential_dmu[1]) / (dT ** 2.0)
	heat_capacity = T * (grandPotential_dT[0] - 2.0 * grandPotential_dT[1] + grandPotential_dT[2]) / (dT ** 2.0)
	return coverage, entropy, susceptibility, heat_capacity, grandPotential_dT[1]
