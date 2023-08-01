import numpy as np 
import math
from tensor import Tensor

def softmax(tensor: Tensor):
    row = []
    sum_of_all = sum([math.exp(x) for x in tensor])
    for value in tensor:
        row.append(math.exp(value) / sum_of_all)
    return Tensor(row)


def tanh(tensor):
    negative_tensor = [-value for value in tensor]
    upper = (exp(tensor) - exp(negative_tensor))
    down = (exp(tensor) + exp(negative_tensor))
    res = upper / down
    return res


def exp(tensor: Tensor):
    res = []
    for value in tensor:
        res.append(math.exp(value))
    return Tensor(res)


def relu(tensor: Tensor):
    res = []
    for value in tensor:
        if value <= 0:
            res.append(0)
            continue
        res.append(value)
    return res