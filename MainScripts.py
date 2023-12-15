from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import TensorNetworks as tn
import BuildTensors as bt

class CalcConfig:

	def __init__(self, methodTolerance = 1e-8, constant = 0.008314, method = "trg", metModification = "default", scale = 4, iterations = 300, model = "ising", lattice = "square", gen_tensor = "default", nodes = 1 , coord = 4, metParam = 10):
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

	def __str__(self):
		return method + "_p_" + str(metParam) + "_" + model + "_" + lattice

def simulate(calc, T = 1.0, m_par = [0.0] * 10):

	if calc.lattice == "triangular":
		calc.coord = 6

	matrixes = bt.build_matrix(calc, T, m_par)
	if calc.method != "htn":
		tensors = tn.build_tensor(calc, matrixes)

	scale = 0.0
	old_scale = -1.0
	norm = 0

	for i in range(calc.iterations):
		if calc.method == "trg":
			(tensors, scale, norm) = tn.trg_step(tensors, scale, norm, calc)
		elif calc.method == "hotrg":
			(tensors, scale, norm) = tn.hotrg_step(tensors, scale, norm, calc)
		elif calc.method == "htn":
			(tensors, scale, norm) = tn.htn_step(matrixes, scale, norm, calc)
		else:
			assert False, "Error! There is no such method."
		if abs(old_scale - scale / calc.scale) < calc.methodTolerance:
			break
		else:
			old_scale = scale
	if i > 250:
		print("Warning! More than 250 iterations")
	nodes = calc.nodes
	nodes *= calc.scale ** (i + 1)
	return (scale + log(norm)) / (nodes / (calc.constant * T))

def coverage_old(method, model, lattice, temp = 1., m_par = [0.0]*10):
	result = derivative(lambda x: simulate(method, model, lattice, temp, [x]+m_par[1:]), m_par[0], n=1, dx=1e-3)
	return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
	return result

def heat_capasity(calc, T = 1., m_par = [0.0]*10):
	result = T * derivative(lambda x: simulate(calc, x, m_par), T, n=2, dx=1e-5)
	return result

def enthropy_old(method,model,size, temp = 1., field = 0, int = [0]):
	result = derivative(lambda x: simulate(model,size,x,field,int), temp, n=1, dx=1e-5)
	return result

def coverage(method, model, lattice, temp = 1., m_par = [0.0]*10, temp_size = 300):
	BTP = []
	mu_step = 1e-3
	for mu_TRG in [m_par[0] - mu_step, m_par[0] + mu_step]:
		lnZ = simulate(method, model, lattice, temp, [mu_TRG] + m_par[1:])
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(mu_step*2.0)
	return result

def coverage_h(func, model, T = 1.0, m_par = [0.0]*10, size = 1):
	result = derivative(lambda x: func(model, T, [x] + m_par[1:], size), m_par[0], n=1, dx=0.001)
	return result

def entropy(method, model, lattice, temp = 1., m_par = [0.0]*10):
	BTP = []
	temp_step = 1e-3
	for temp_TRG in [temp - temp_step, temp + temp_step]:
		lnZ = simulate(method, model, lattice, temp_TRG, m_par)
		BTP.append(lnZ)

	result = -(BTP[0]-BTP[1])/(temp_step*2.0)
	return result

def full(calc, T = 1., m_par = [0.0] * 10, dmu = 1e-3, dT = 1e-3, derivatives = [1, ] + [0] * 10):
	grandPotential_dmu = []
	grandPotential_dT = []
	for diff_mu in [m_par[0] - dmu, m_par[0] + dmu]:
		lnZ = simulate(calc, T, [diff_mu] + m_par[1:])
		grandPotential_dmu.append(lnZ)
	for diff_T in [T - dT, T, T + dT]:
		lnZ = simulate(calc, diff_T, m_par)
		grandPotential_dT.append(lnZ)
	coverage = - (grandPotential_dmu[0] - grandPotential_dmu[1]) / (dmu * 2.0)
	entropy = - (grandPotential_dT[0] - grandPotential_dT[2]) / (dT * 2.0)
	susceptibility = calc.constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dT[1] + grandPotential_dmu[1]) / (dT ** 2.0)
	heat_capacity = T * (grandPotential_dT[0] - 2.0 * grandPotential_dT[1] + grandPotential_dT[2]) / (dT ** 2.0)
	return coverage, entropy, susceptibility, heat_capacity, grandPotential_dT[1]
