import numpy as np 
import math
from tensor import Tensor
from utils import zeros_like, sum_tensor, exp

#Helper functions
def _softmax_tensor(tensor):
    row = []
    sum_of_all = sum([math.exp(x) for x in tensor])
    for value in tensor:
        row.append(math.exp(value) / sum_of_all)
    return Tensor(row)

def _softmax_matrix(matrix, dim):
    rows, cols = matrix.shape
    exp_fn = exp(matrix)
    output = zeros_like(matrix)

    if isinstance(dim ,int):
        if dim == 0:
            sum_arr = sum_tensor(exp_fn, 0)
            for i in range(rows):
                for j in range(cols):
                    output[j][i] = exp(matrix[j][i]) / sum_arr[i]
                    if np.isnan(output[j][i]):
                        output[j][i] = 1
        else:
            sum_arr = sum_tensor(exp_fn, 1)
            for i in range(rows):
                for j in range(cols):
                    output[i][j] = exp(matrix[i][j]) / sum_arr[i]
                    if np.isnan(output[j][i]):
                        output[i][j] = 1
    return output


def softmax(tensor: Tensor, dim=None):
    if dim != None:
        res = _softmax_matrix(tensor, dim)
        return res
    else:
        res = _softmax_tensor(tensor)
        return res 




def tanh(tensor):
    unpacked = []
    for value in tensor:
        if isinstance(value, list):
            for val2 in value:
                unpacked.append(val2)
        else:
            unpacked.append(value)
    
    negative_tensor = [-value for value in unpacked]
    upper = (exp(unpacked) - exp(negative_tensor))
    down = (exp(unpacked) + exp(negative_tensor))
    res = upper / down
    return Tensor(res)


def relu(tensor: Tensor):
    res = []
    for value in tensor:
        if value <= 0:
            res.append(0)
            continue
        res.append(value)
    return Tensor(res)