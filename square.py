from math import exp
import numpy as np

def get_tensor_2 (model, beta = 1., field = 0, int1 = 0):

    tensor_dict = {
        "langmuir" : np.array([[0.0, field/2.0],[field/2.0, -int1+field]]),
        "ising" : np.array([[(int1-field), (-int1)],[(-int1), (int1+field)]]),
        "hard-disk" : np.array([[0.0, field/2.0],[field/2.0, -1000000.0+field]])
    }

    tensor_2 = tensor_dict.get(model)
    assert (tensor_2 is not None), "Error! This model is not in the database"
    tensor_2 *= beta
    tensor_2 = np.array([np.exp(line) for line in tensor_2])
    return tensor_2
