from math import exp, log,sqrt
import numpy as np
from scipy.misc import derivative
import scipy.sparse.linalg
from scipy.linalg import sqrtm
import TensorNetworks as tn
import Build_tensors as bt

constant = 1.
interactions = [1.]
temperature = 1.
l_vector = []
chi_number = 24
chi_min = 1e-8
method_tolerance = 1e-8

def simulate(method = "trg", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0]*10, temp_size = 300):

    tensors = bt.build_matrix(model, temp, m_par, 6.0)
    tensors = tn.build_tensor(tensors, lattice)
    #tensors = tn.build_triangles_tensor(model, temp, m_par)

    scale = 0.0
    old_scale = -1.0
    if lattice == "triangle":
        nodes = 1.0
    else:
        nodes = 2.0

    i = 0
    for i in range(temp_size):
        if method == "trg":
            (tensors, scale) = tn.trg_step(tensors, scale, chi_number, chi_min, lattice)
        elif method == "hotrg":
            (tensors, scale) = tn.hotrg_step(tensors, scale, chi_number, chi_min, lattice)
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
    return (scale+log(norm))/(nodes/(constant*temp))

def simulate2(method = "trg", model = "langmuir", lattice = "square", temp = 1.0, m_par = [0.0]*10, temp_size = 300):
    res = []
    for test_chi in range(35, 46):
        print(test_chi)
        res.append(simulate(method, model, lattice, temp, m_par, temp_size))
    temp = []
    for i in range(len(res)):
        if (i > 0) and (i < len(res) - 1):
            temp.append(abs(res[i] - res[i-1] + res[i+1] - res[i]) / res[i])
    result = temp.index(min(temp))
    return result

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

def full(method, model, lattice, temp = 1., m_par = [0.0]*10, temp_size = 300):
    BTP_mu = []
    BTP_temp = []
    step = 0.1
    for mu_TRG in [m_par[0] - step, m_par[0] + step]:
        lnZ = simulate2(method, model, lattice, temp, [mu_TRG] + m_par[1:], temp_size)
        BTP_mu.append(lnZ)
    for temp_TRG in [temp - step, temp, temp + step]:
        lnZ = simulate2(method, model, lattice, temp_TRG, m_par, temp_size)
        BTP_temp.append(lnZ)

    cov = -(BTP_mu[0]-BTP_mu[1])/(step*2.0)
    ent = -(BTP_temp[0]-BTP_temp[2])/(step*2.0)
    sus = (BTP_mu[0]-2.0*BTP_temp[1]+BTP_mu[1])/(step**2.0)
    cap = (BTP_temp[0]-2.0*BTP_temp[1]+BTP_temp[2])/(step**2.0)
    return cov, ent, sus, cap

method = "trg"
model = "4NN_triangular"
lattice = "tr_to_sqr"
temp_square = 2.0/log(1+sqrt(2))
#temp_hex = 4.0/log(3)
#temp = temp_hex
#temp = 4.0/log(3) + 1e-7
temp = 1.0
mu = 1.0
chi_number = 100
system_size = 300
#for system_size in np.arange(2, 50, 1):
#   print("system_size=", system_size)
    #m_par = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
for chi_number in np.arange(24, 2000, 1000):
    #print("xhi=", chi_number)
    #   print(chi_number, 2.0*simulate(method, model, lattice, temp, m_par))
    #for temp in np.arange(2.0, 4.41, 0.05):
    for mu in np.arange(6.0, -6.01, -0.1):
        m_par = [mu, 0.0, 0.0, 0.0, 0.0, 0.0]
        #m_par = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        coef = 2.0/1.00
        result = full(method, model, lattice, temp, m_par, system_size)
        print(mu, coef*result[0], coef*result[1], coef*result[2], coef*result[3], chi_number)
        #print(chi_number, simulate(method, model, lattice, temp, m_par, chi_number))
        #result = coverage(method, model, lattice, temp, m_par, system_size)
        #print(mu, coef*result, chi_number)
