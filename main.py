from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import TensorNetworks as tn
import Build_tensors as bt

constant = 1.
method_tolerance = 1e-8

def simulate(method = "trg", model = "langmuir", lattice = "square", T = 1.0, m_par = [0.0]*10, chi_number = 300):

    tensors = bt.build_matrix(model, T, m_par, 6.0)
    tensors = tn.build_tensor(tensors, lattice)
    #tensors = tn.build_triangles_tensor(model, temp, m_par)

    scale = 0.0
    old_scale = -1.0
    if lattice == "triangle":
        nodes = 1.0
    else:
        nodes = 2.0

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
    return (scale+log(norm))/(nodes/(constant*T))

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
    dmu = 0.01
    dT = 0.01
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
    return coverage, entropy, susceptibility, heat_capacity

method = "trg"
model = "qstate"
lattice = "triangular"
#model params
c = 1
n = 6
epsilon = -1
delta = 1.0
T = 0.05
chi_number = 40
for mu in np.arange(-1.00, 7.01, 0.2):
	m_par = [mu, c, n, epsilon, delta, 0.0]
	coef = 2.0/1.00
	result = full(method, model, lattice, chi_number, T, m_par)
	print(mu, coef*result[0], coef*result[1] , coef*result[2] , coef*result[3])
