from math import exp, log
import numpy as np
from scipy.misc import derivative

def get_simple_tensor_2 (model, beta = 1., field = 0, int = 0):

    tensor_dict = {
        "langmuir" : np.array([[0.0, field/2.0],[field/2.0, -int+field]]),
        "ising" : np.array([[(int-field), (-int)],[(-int), (int+field)]]),
        "hard-disk" : np.array([[0.0, field/2.0],[field/2.0, -1000000.0+field]])
    }

    tensor_2 = tensor_dict.get(model)
    assert (tensor_2 is not None), "Error! This model is not in the database"
    tensor_2 *= beta
    tensor_2 = np.array([np.exp(line) for line in tensor_2])
    return tensor_2

def create_cd(dim,elem):
    cd = np.zeros((elem,) * dim)
    for i in range(elem):
        cd[((i,)*dim)] = 1.
    return cd

interactions = 5.
temperature = 200.

def eigenvalues(size, hor_tensor_2, vert_tensor_2, tolerance = 1e-10, vectors = False):
    elem = hor_tensor_2.shape[0]
    l_vector = np.random.rand(elem**size).reshape([elem for i in range(size)])
    dop_vector = l_vector + 1e-1
    vert_tensor_2 = np.einsum("kl,ijk,lmn->ijmn",vert_tensor_2,create_cd(3,elem),create_cd(3,elem))
    error = 1.0
    eigenvalue = None
    while error > tolerance:
        for i in range(size):
            l_vector = np.tensordot(l_vector,hor_tensor_2, axes=([0],[0]))
        for i in range(size):
            l_vector = np.tensordot(l_vector,vert_tensor_2, axes=([0,-1],[0,2]))
        eigenvalue = np.linalg.norm(l_vector)
        l_vector /= eigenvalue
        error = np.linalg.norm(l_vector - dop_vector)
        print(error)
        l_vector = 0.5*(l_vector) + 0.5*(dop_vector)
        dop_vector = l_vector
    return eigenvalue

def simple_TM(model,size, beta = 1., field = 0, int = 0):
    tensor = get_simple_tensor_2(model, beta,field, int = interactions)
    return log(eigenvalues(size, tensor, tensor))/beta/(2.*size)

def coverage(model,size, beta = 1., field = 0, int = 0):
	result = derivative(lambda x: simple_TM(model,size,beta,x,int), field, n=1, dx=0.001)
	return result

for muu in np.arange(5,5.1,0.2):
    print(muu,coverage('langmuir',8, 1./(0.008314*temperature), muu, interactions))
