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
	#1D long-range models build their transfer matrix directly in build_matrix
	#(a single ready matrix on a chain with coordination number 2), so they bypass
	#build_tensor and feed the raw matrix straight into tm_step; every other model
	#builds a tensor first
	tm_uses_raw_matrix = (calc.method == "tm" and calc.model == "1D_long-range")
	if tm_uses_raw_matrix:
		calc.coord = 2

	matrixes, first_norm = bt.build_matrix(calc, T, m_par)
	if calc.method != "htn" and not tm_uses_raw_matrix:
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
			if tm_uses_raw_matrix:
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

def thermodynamics(calc, T = 1.0, m_par = None, *, coverage = False, susceptibility = False,
		entropy = False, heat_capacity = False, mu_index = 0, dmu = 1e-3, dT = 1e-3):
	"""Compute the requested thermodynamic observables.

	Each observable is a finite-difference derivative of the grand potential
	Omega = simulate(calc, T, m_par).  Only the points needed for the requested
	observables are evaluated, and the central point Omega(mu0, T0) is shared
	between the two second derivatives:

	    coverage        first,  d/dmu     -> mu-, mu+
	    susceptibility  second, d2/dmu2   -> mu-, center, mu+
	    entropy         first,  d/dT      -> T-, T+
	    heat_capacity   second, d2/dT2    -> T-, center, T+

	mu_index selects which chemical potential in m_par is differentiated; it may
	be a single index or a list of indices that are shifted together (the total
	coverage / susceptibility with respect to several chemical potentials, which
	the multi-component adsorption models need).

	Returns a dict mapping each requested observable to its value; grand_potential
	(= Omega(mu0, T0)) is included whenever a second derivative is requested.
	"""
	if m_par is None:
		m_par = [0.0] * 10

	if isinstance(mu_index, int):
		mu_indices = [mu_index]
	else:
		mu_indices = list(mu_index)

	need_mu = coverage or susceptibility
	need_T = entropy or heat_capacity
	need_center = susceptibility or heat_capacity

	result = {}

	center = None
	if need_center:
		center = simulate(calc, T, m_par)
		result["grand_potential"] = center

	if need_mu:
		mu_minus = m_par[:]
		mu_plus = m_par[:]
		for j in mu_indices:
			mu_minus[j] -= dmu
			mu_plus[j] += dmu
		omega_mu_minus = simulate(calc, T, mu_minus)
		omega_mu_plus = simulate(calc, T, mu_plus)
		if coverage:
			result["coverage"] = -(omega_mu_minus - omega_mu_plus) / (2.0 * dmu)
		if susceptibility:
			result["susceptibility"] = calc.constant * T * (omega_mu_minus - 2.0 * center + omega_mu_plus) / (dmu ** 2)

	if need_T:
		omega_T_minus = simulate(calc, T - dT, m_par)
		omega_T_plus = simulate(calc, T + dT, m_par)
		if entropy:
			result["entropy"] = -(omega_T_minus - omega_T_plus) / (2.0 * dT)
		if heat_capacity:
			result["heat_capacity"] = T * (omega_T_minus - 2.0 * center + omega_T_plus) / (dT ** 2)

	return result

def _mu_mask_to_indices(derivatives):
	#convert a 0/1 mask like [1, 1] into the list of indices [0, 1]
	return [i for i, flag in enumerate(derivatives) if flag == 1]

def heat_capasity(calc, T = 1., m_par = [0.0] * 10, dT = 1e-4):
	#thin wrapper over thermodynamics(); dT = 1e-4 reproduces the legacy
	#scipy.misc.derivative(..., n = 2, dx = 1e-4) used before SciPy 1.12
	return thermodynamics(calc, T, m_par, heat_capacity = True, dT = dT)["heat_capacity"]

def susceptibility(calc, T = 1., m_par = [0.0] * 10, dmu = 1e-4, derivatives = [1, ] + [0] * 2):
	#thin wrapper over thermodynamics()
	return thermodynamics(calc, T, m_par, susceptibility = True,
		mu_index = _mu_mask_to_indices(derivatives), dmu = dmu)["susceptibility"]

def full(calc, T = 1., m_par = [0.0] * 10, dmu = 1e-3, dT = 1e-3, derivatives = [1, ] + [0] * 2, T_derivative = True, mu_derivative = True):
	#backward-compatible wrapper returning the legacy tuple
	#(coverage, entropy, susceptibility, heat_capacity, grand_potential).
	#derivatives is a 0/1 mask selecting which chemical potentials are
	#differentiated together.  susceptibility is only requested when T_derivative
	#is also on, so the central point is reused and the simulate() call count
	#(and coverage) match the previous implementation exactly.
	obs = thermodynamics(calc, T, m_par,
		coverage = mu_derivative, susceptibility = mu_derivative and T_derivative,
		entropy = T_derivative, heat_capacity = T_derivative,
		mu_index = _mu_mask_to_indices(derivatives), dmu = dmu, dT = dT)
	return (obs.get("coverage", 0.0), obs.get("entropy", 0.0),
		obs.get("susceptibility", 0.0), obs.get("heat_capacity", 0.0),
		obs.get("grand_potential", 0.0))
