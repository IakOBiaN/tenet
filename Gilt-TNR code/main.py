from math import exp, log, sqrt
import numpy as np
from scipy.misc import derivative
import GiltTNR2D as gilt
from ncon import ncon
from abeliantensors import Tensor
import sys
from scipy.linalg import sqrtm
import timeit

model_name = 'GiltTNR'

sys.setrecursionlimit(5000)

constant = 1.0

def create_cd(dim,elem):
	cd = np.zeros((elem,) * dim)
	for i in range(elem):
		cd[((i,)*dim)] = 1.
	return cd

def build_tensor(model, lattice, temp, m_par):
	inf = -1e8
	if (lattice == "square"):
		neigbours = 4.0
	elif (lattice == "triangle"):
		neigbours = 6.0

	if model == "langmuir":
		matrix = np.array([[0.0, m_par[0]/neigbours],[m_par[0]/neigbours, -m_par[1]+m_par[0]/(neigbours/2.0)]])
	elif model == "ising":
		matrix = np.array([[(m_par[1]-m_par[0]/(neigbours/2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1]+m_par[0]/(neigbours/2.0))]])
	elif model == "hard-disk":
		matrix = np.array([[0.0, m_par[0] / (neigbours)],[m_par[0] / (neigbours), inf + m_par[0]]])

	matrix *= 1./(constant*temp)
	matrix = np.array([np.exp(line) for line in matrix])
	n = matrix.shape[0]
	cd = create_cd(3, n)
	tensor = np.einsum("ab,ibk->iak",matrix,cd)
	if (lattice == "square"):
		tensor = np.einsum("abc,bef,ejk,jcn->afkn",tensor,tensor,tensor,tensor)
		#cd = create_cd(4, n)
		#matrix = sqrtm(matrix)
		#tensor = np.einsum("abcd,ia,bl,cn,od->ilnd", cd, matrix, matrix, matrix, matrix)
	elif (lattice == "triangle"):
		tensor = np.einsum("abc,dfb,icf->adi",tensor,tensor,tensor)
		tensor = np.einsum("abi,ijk->abjk", tensor, cd)

	"""if model == "hard_triangles":
		field = m_par[0]/6.0
		one = np.array([[0, inf, field, field, field, inf, field], \
						[field, inf, 2.0*field, 2.0*field, 2.0*field, inf, 2.0*field], \
						[inf, 2.0*field, inf, inf, inf, inf, inf], \
						[field, inf, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[field, inf, 2.0*field, 2.0*field, inf, inf, 2.0*field], \
						[field, inf, 2.0*field, 2.0*field, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf]])
		two = np.array([[0, field, field, inf, field, inf, field], \
						[field, inf, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[inf, inf, inf, inf, inf, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf]])
		three = np.array([[0, field, field, inf, field, field, inf], \
						[inf, inf, inf, 2.0*field, inf, inf, inf], \
						[field, 2.0*field, inf, inf, 2.0*field, 2.0*field, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, 2.0*field, inf], \
						[inf, inf, inf, inf, inf, inf, 2.0*field], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, inf, inf], \
						[field, 2.0*field, 2.0*field, inf, 2.0*field, 2.0*field, inf]])

		one *= 1./(constant*temp)
		one = np.array([np.exp(line) for line in one])
		two *= 1./(constant*temp)
		two2 = np.array([np.exp(line) for line in two])
		three *= 1./(constant*temp)
		three = np.array([np.exp(line) for line in three])
		cd = create_cd(3, one.shape[0])
		one = np.einsum("ab,iak->ibk",one,cd)
		two = np.einsum("ab,iak->ibk",three,cd)
		three = np.einsum("ab,ibk->iak",two2,cd)
		tensor = np.einsum("abc,deb,ice->adi",one,two,three)
		tensor = np.einsum("abi,ijk->abjk", tensor,cd)"""

	tensor = Tensor.from_ndarray(tensor)
	return tensor

pars = {
    "gilt_eps": 8e-7,
	"gilt_print_envspec": False,
	"gilt_print_envspec_recursive": False,
	"cg_chis": 10,
	"print_spectra": False,
	"cg_eps": 1e-8,
	"verbosity": 0,
	"debug": False
}

def simulate(method = "G_tnr", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0, 0.0]):
	A = build_tensor(model, lattice, temp, m_par)
	log = 0.0
	old_log = -1.0
	if lattice == "square":
		nodes = 2.0
	elif lattice == "triangle":
		nodes = 1.0
	for i in range(100):
		#print(i)
		if method == "G_tnr":
			(A, log) = gilt.gilttnr_step(A, log, pars)
		elif method == "trg":
			(A, log) = gilt.trg_step(A, log, pars)
		else:
			assert False, "Error! There is no such method."
		if abs(old_log - log/4.0) < 1e-9:
			break
		else:
			old_log = log
	if i > 50:
		pritn("Warning! More than 50 iterations")
	nodes *= 4**(i+1)

	return log/(nodes/(constant*temp))# - 0.5*temp

def coverage(method = "G_tnr", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0, 0.0]):
	result = derivative(lambda x: simulate(method, model, lattice, temp, [x, m_par[1]]), m_par[0], n=1, dx=1e-3)
	return result

def heat_capasity(method = "G_tnr", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0, 0.0]):
	result = derivative(lambda x: simulate(method, model, lattice, x, m_par), temp, n=2, dx=1e-5)
	return result

def full(method, model, lattice, T, m_par = [0.0]*10):
	grandPotential_dmu = []
	grandPotential_dT = []
	dmu = 1e-5
	dT = 1e-5
	for diff_mu in [m_par[0] - dmu, m_par[0] + dmu]:
		lnZ = simulate(method, model, lattice, T, [diff_mu] + m_par[1:])
		grandPotential_dmu.append(lnZ)
	for diff_T in [T - dT, T, T + dT]:
		lnZ = simulate(method, model, lattice, diff_T, m_par)
		grandPotential_dT.append(lnZ)
	coverage = - (grandPotential_dmu[0] - grandPotential_dmu[1]) / (dmu * 2.0)
	entropy = - (grandPotential_dT[0] - grandPotential_dT[2]) / (dT * 2.0)
	susceptibility = constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dT[1] + grandPotential_dmu[1]) / (dT ** 2.0)
	heat_capacity = T * (grandPotential_dT[0] - 2.0 * grandPotential_dT[1] + grandPotential_dT[2]) / (dT ** 2.0)
	return coverage, entropy, susceptibility, heat_capacity, grandPotential_dT[1]

method = "G_tnr"
model = "hard-disk"
lattice = "triangle"
temp = 1.0
for mu in np.arange(-6.0,6.0,0.1):
	m_par = [mu, 0.0]
	print(mu, coverage(method, model, lattice, temp, m_par))

invest = [i for i in range(2, 80, 1)]

for lat_size in invest:
	pars["cg_chis"] = size
	f = open('results_' + model_name + str(lat_size) + '.dat', 'w')
	f2 = open('time_' + model_name + str(lat_size) + '.dat', 'w')
	f.write('mu	coverage S sus Cp calc_time''\n')

	start_time = timeit.default_timer()
	calc_time = 0

	for mu in ms.np.arange(-6.00, 6.01, 0.1):
		m_par = [mu, 0.0, 0, 0, 0, 0]
		result = full(method, model, lattice, temp, m_par)
		calc_time = timeit.default_timer() - start_time
		print(mu, result[0], calc_time, lat_size)
		f.write(('{}\t' * 6).format(mu, result[0], result[1], result[2], result[3], calc_time) + '\n')
		f2.write(str(T) + " " + str(mu) + " " + str(calc_time) + "\n")

	f2.write("Res. calculation_time: " + str(calc_time) + "\n")
	f.close()
	f2.close()
