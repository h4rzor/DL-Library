import numpy as np 
import matplotlib.pyplot as plt 
import math
import torch
import random


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __str__(self):
        return f"Tensor({self.data})"

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        if isinstance(self.data, (int, float)):
            return 0
        elif isinstance(self.data, list) or isinstance(self.data, np.ndarray):
            if isinstance(self.data[0], (int, float)):
                return (len(self.data), 0)
            elif all(isinstance(row, list) or (isinstance(row, np.ndarray)) for row in self.data):
                return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row_index, col_index = index
            if isinstance(row_index, int) and isinstance(col_index, int):
                return self.data[row_index][col_index]

            elif isinstance(row_index, slice) and isinstance(col_index, slice):
                if row_index.start is None and row_index.stop is None and row_index.step is None \
                        and col_index.start is None and col_index.stop is None and col_index.step is None:
                            return self.data[:]
                else:
                    return [row[col_index] for row in self.data[row_index]]

            elif isinstance(row_index, int) and isinstance(col_index, slice):
                return self.data[row_index][col_index]
            elif isinstance(row_index, slice) and isinstance(col_index, int):
                return [row[col_index] for row in self.data[row_index]]
        else:
            return self.data[index]

    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __add__(self, other):
        if isinstance(self, int):
            return Tensor(self + other.data, (self, other), "+")
        elif isinstance(other, int):
            return Tensor(self.data + other, (self, other), "+")
        elif isinstance(self.data, int) and isinstance(other.data, int):
            return Tensor(self.data + other.data, (self, other), "+")

        self_res = all(isinstance(ele, list) for ele in self.data)
        other_res = all(isinstance(ele, list) for ele in other.data)
        if not self_res and not other_res and len(self.data) == len(other.data):
            return Tensor(np.array(self.data) * np.array(other.data), (self, other), "+")
        
        if len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
            return Tensor(np.add(self.data, other.data), (self, other), "+")
            
    
    def __mul__(self, other):
        if isinstance(self.data, int) and isinstance(other.data, int):
            return self.data * other.data

        elif isinstance(other, int):
            return list([elem * other for elem in self.data])
        
        elif isinstance(self, int):
            return list([elem * self for elem in other.data]) 


        self_res = all(isinstance(ele, list) for ele in self.data)
        other_res = all(isinstance(ele, list) for ele in other.data)
        if not self_res and not other_res and len(self.data) == len(other.data):
            return Tensor(np.array(self.data) * np.array(other.data))

        if len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
            return Tensor(np.multiply(self.data, other.data), (self, other), "*")

    def __matmul__(self, other):
        if self.shape[1] == other.shape[0]:
            return Tensor(np.matmul(self.data, other.data))

    def transpose(self):
        return np.transpose(self)

    def __sub__(self, other):
        if isinstance(other, int):
            return list([elem - other for elem in self.data])

        self_res = all(isinstance(ele, list) for ele in self.data)
        other_res = all(isinstance(ele, list) for ele in other.data)
        
        if isinstance(self.data, int) and isinstance(other.data, int):
            return self.data - other.data

        if not self_res and not other_res:
            return Tensor(np.subtract(self.data, other.data))
        else:
            row1, col1 = self.shape
            row2, col2 = other.shape

            if row1 == row2 and col1 == col2:
                return Tensor(np.subtract(self.data, other.data))
    
    def __rsub__(self, other):
        if isinstance(other, int):
            return list([elem - other for elem in self.data])

    def __truediv__(self, other):
        if isinstance(self.data, int) and isinstance(other.data, int):
            return Tensor(self.data / other.data)
        
        self_res = all(isinstance(ele, list) for ele in self.data)
        other_res = all(isinstance(ele, list) for ele in other.data)

        if not self_res and not other_res:
            return Tensor(np.divide(self.data, other.data))
        else:
            row1, col1 = self.shape
            row2, col2 = other.shape

            if row1 == row2 and col1 == col2:
                return Tensor(np.divide(self.data, other.data))
    
    def __iadd__(self, other):
        if isinstance(other, Tensor):
            self.data += other.data
        elif isinstance(other, (int, float)):
            self.data += other
        return self
    
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data + other, (self, other), "+")