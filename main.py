from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import TensorNetworks as tn
import Build_tensors as bt

constant = 0.008314
method_tolerance = 1e-8

def calc(method = "trg", model = "langmuir", lattice = "square", T = 1.0, m_par = [0.0]*10, chi_number = 300):

    tensors = bt.build_matrix(model, T, m_par, 4.0)
    tensors = tn.build_tensor(tensors, lattice)
    #tensors = tn.build_triangles_tensor(model, temp, m_par)

    scale = 0.0
    old_scale = -1.0
    nodes = 2.0
    if lattice == "triangular":
        nodes = 1.0

    i = 0
    for i in range(300):
        if method == "trg":
            (tensors, scale) = tn.trg_step(tensors, scale, chi_number, 0, lattice)
        elif method == "hotrg":
            (tensors, scale) = tn.hotrg_step(tensors, scale, chi_number, 0, lattice)
        else:
            assert False, "Error! There is no such method."
        if abs(old_scale - scale/4.0) < method_tolerance:
            break
        else:
            old_scale = scale
    if i > 250:
        print("Warning! More than 250 iterations")
    nodes *= 4.0**(i+1)
    norm = np.einsum("abab->",tensors[0])
    if norm < 0:
        norm = -norm
    return (scale + log(norm)) / (nodes / (constant * T))

def simple_hierarchical(method, model, lattice, T = 1.0, m_par = [0.0]*10, size = 1):
    tensor = bt.build_matrix(model, T, m_par, 4.0)[0]
    number_of_steps = 100

    Z = np.empty((number_of_steps+1))
    Z[0] = tensor.max()
    tensor = tensor/Z[0]
    lnZ_list = []
    cd3 = tn.identity(3, tensor.shape[0])
    cd4 = tn.identity(4, tensor.shape[0])

    for i in np.arange(1,(number_of_steps+1),1):
        edges = (1 + size * 2) ** 2
        dop_tensor = tn.identity(size+2,tensor.shape[0])
        dop_tensor_2 = tn.identity(size+2,tensor.shape[0])

        doubled_tensor = np.einsum("ij,aic->ajc",tensor,cd3)
        doubled_tensor = np.einsum("ij,abi->abj",tensor,doubled_tensor)
        doubled_tensor = np.einsum("ijk,ajk->ai",doubled_tensor,cd3)

        for ii in range(size):
            dop_tensor = np.tensordot(dop_tensor,doubled_tensor, axes=([1],[0]))
        dop_tensor = np.tensordot(dop_tensor,tensor, axes=([1],[0]))

        for ii in range(size+(size-1)):
            dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-1],[0]))
            for j in range(size):
                dop_tensor = np.tensordot(dop_tensor,cd3, axes=([-2-j],[0]))
                dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-2],[0]))
                dop_tensor = np.tensordot(dop_tensor,cd3, axes=([-1,-3],[0,1]))
                dop_tensor = np.tensordot(dop_tensor,tensor, axes=([-2],[0]))

        for ii in range(size):
            dop_tensor = np.tensordot(dop_tensor,doubled_tensor, axes=([1],[0]))
        dop_tensor = np.tensordot(dop_tensor,tensor, axes=([1],[0]))
        tensor = np.tensordot(dop_tensor,dop_tensor_2, axes=(np.arange(1,size+2,1),np.arange(1,size+2,1)))

        #if tensor.max() > 1e10:
        #    Z[i] = tensor.max()
        #else:
        Z[i] = tensor.max()#np.trace(tensor)
        #if Z[i] > 1000:
        #    Z[i] *= np.sqrt(tensor.max())
        #Z[i] = np.trace(tensor)#tensor.max()
        tensor = tensor/Z[i]

        xxx = np.trace(tensor)
        #print(i, xxx)
        if xxx < 1e-10:
            break

        lnZ = np.log(xxx)
        lnZ /= (edges ** (i+2))
        logZ_powers_sum = sum(sorted([log(Z[j]) / (edges**j) for j in range(0,i+1)]))
        lnZ += logZ_powers_sum

        lnZ_list.append(lnZ)
        if len(lnZ_list) > 3 and abs(lnZ_list[-1] - lnZ_list[-2]) < 1e-9 and abs(lnZ_list[-1] - lnZ_list[-3]) < 1e-9:
            break
    beta = 1 / (constant * T)
    return (lnZ_list[-1]) / beta * 2.0

def coverage_old(method, model, lattice, temp = 1., m_par = [0.0]*10):
    result = derivative(lambda x: simulate(method, model, lattice, temp, [x]+m_par[1:]), m_par[0], n=1, dx=1e-3)
    return result

def magnetization(method,model,size, temp = 1., field = 0, int = [0]):
    result = derivative(lambda x: simulate(model,size,temp,field,[x]), int[0], n=1, dx=1e-5)
    return result

def heat_capasity(method, model, lattice, temp = 1., m_par = [0.0]*10):
    result = derivative(lambda x: simulate(method, model, lattice, x, m_par), temp, n=2, dx=1e-3)
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

def full(method, model, lattice, chi_number, T = 1., m_par = [0.0]*10):
    grandPotential_dmu = []
    grandPotential_dT = []
    dmu = 0.001
    dT = 0.001
    if method == "hierarchical":
        simulate = simple_hierarchical
    else:
        simulate = calc
    for diff_mu in [m_par[0] - dmu, m_par[0] + dmu]:
        lnZ = simulate(method, model, lattice, T, [diff_mu] + m_par[1:], chi_number)
        grandPotential_dmu.append(lnZ)
    for diff_T in [T - dT, T, T + dT]:
        lnZ = simulate(method, model, lattice, diff_T, m_par, chi_number)
        grandPotential_dT.append(lnZ)
    coverage = - (grandPotential_dmu[0] - grandPotential_dmu[1]) / (dmu * 2.0)
    entropy = - (grandPotential_dT[0] - grandPotential_dT[2]) / (dT * 2.0)
    susceptibility = constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dT[1] + grandPotential_dmu[1]) / (dT ** 2.0)
    heat_capacity = T * (grandPotential_dT[0] - 2.0 * grandPotential_dT[1] + grandPotential_dT[2]) / (dT ** 2.0)
    return coverage, entropy, susceptibility, heat_capacity, grandPotential_dT[1]

#qstate_model
"""method = "trg"
model = "qstate"
lattice = "triangular"
#model params
c = 1
n = 12
epsilon = -1
delta = 1.0
T = 0.05
chi_number = 60
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
	print(mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3])"""

#langmuir
"""
method = "hierarchical"
model = "langmuir"
lattice = "FSHL"
#model params
T = 1.0
chi_number = 5
for mu in np.arange(-10.00, 25.01, 0.5):
	m_par = [mu, 4.0, 0, 0, 0, 0]
	coef = 1.0/1.00
	result = full(method, model, lattice, chi_number, T, m_par)
	print(mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3])
"""


method = "hierarchical"
model = "binary"
lattice = "FSHL"
#model params
T = 100.0
chi_number = 1
for mu in np.arange(-40.00, 30.01, 1.0):
	m_par = [mu, 10.0, 4.0, 6.0, 0, 0]
	result = full(method, model, lattice, chi_number, T, m_par)
	print(mu, result[0], result[1] , result[2] , result[3])
