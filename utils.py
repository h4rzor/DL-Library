import math 
import numpy as np
import random
from typing import Tuple
from tensor import Tensor

# Helper functions
def _make_matrix(rows, cols, value):
    matrix = []

    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(value)
        matrix.append(row)
    return matrix

def _exp_scalar(tensor):
    return math.exp(tensor)

def _exp_vector(tensor):
    res = []
    for value in tensor:
        if isinstance(value, list):
            for el in value:
                res.append(math.exp(el))
            return res
        else:
            res.append(math.exp(value))
    return Tensor(res)

def _exp_matrix(tensor: Tensor):
    rows, cols = tensor.shape
    res = zeros_like(tensor)
    for i in range(rows):
        for j in range(cols):
            res[i][j] = exp(tensor[i][j])
    return res

#Functions

def zeros(shape: Tuple[int, int]) -> "Tensor":
    rows, cols = shape
    return Tensor(_make_matrix(rows, cols, 0))

def zeros_like(tensor: Tensor):
    rows, cols = tensor.shape
    return Tensor(_make_matrix(rows, cols, 0))

def ones(shape: Tuple[int, int]) -> "Tensor":
    rows, cols = shape
    return Tensor(_make_matrix(rows, cols, 1))

def ones_like(tensor: Tensor):
    if tensor.shape[1] == 0:
        return [1 for _ in range(tensor.shape[0])]

    rows, cols = tensor.shape
    return Tensor(_make_matrix(rows, cols, 1))


def randint(start, end, shape: Tuple[int, int]) -> "Tensor":
    rows, cols = shape
    matrix = zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            value = random.randint(start, end)
            matrix[i][j] = value
    return matrix


def normal(shape: Tuple[int, int]):
    if len(shape) == 1:
        row = []
        for _ in range(shape[0]):
            u1 = random.random()
            u2 = random.random()
            z = ( (-2 * np.log(u1))**0.5 ) * np.cos(2 * np.pi * u2)
            row.append(z)
        return Tensor(row)

    samples = []
    for _ in range(shape[0]):
        row = []
        for _ in range(shape[1]):
            u1 = random.random()
            u2 = random.random()
            z = ( (-2 * np.log(u1))**0.5 ) * np.cos(2 * np.pi * u2)
            row.append(z)
        samples.append(row)
    return Tensor(samples)


def argmax(tensor: Tensor):
    cloned = tensor[:]
    max_number = -math.inf
    index = 0
    if tensor.shape[1] == 0 or tensor.shape[1] == 1:
        for i in range(len(cloned)):
            if isinstance(cloned[i], list) and len(cloned[i]) == 1:
                if cloned[i][0] > max_number:
                    max_number = cloned[i][0]
                    index = i
        return index

def argmin(tensor: Tensor):
    cloned = tensor[:]
    min_number = math.inf
    index = 0
    if tensor.shape[1] == 0:
        for i in range(len(cloned)):
            if cloned[i] < min_number:
                min_number = cloned[i]
                index = i
    return index


def sum_tensor(tensor: Tensor, dim):
    rows, cols = tensor.shape
    if dim == 0:
        sum_cols = []
        for i in range(rows):
            sum_cols.append(sum(tensor[:, i]))
            #the magic happens here :)
        return Tensor(sum_cols)
    else:
        sum_rows = []
        for i in range(cols):
            sum_rows.append(sum(tensor[i, :]))
        return Tensor(sum_rows)


def exp(tensor: Tensor):
    if isinstance(tensor.data, int) or isinstance(tensor.data, int):
        res = _exp_scalar(tensor)
        return res
    elif isinstance(tensor, list) and len(tensor.shape) == 1:
        res = _exp_vector(tensor)
        return res
    else:
        res = _exp_matrix(tensor)
        return res
        