import numpy as np 
import matplotlib.pyplot as plt 
import math
import torch
import random


class Tensor:
    def __init__(self, data):
        self.data = data 

    def __str__(self):
        return f"Tensor({self.data})"

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        rows = len(self.data)
        if rows == 0:
            cols = 0
        else:
            cols = len(self.data[0])
        return rows, cols

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __add__(self, other):
        result = self.isinstance_int(self, other)
        if result:
            return self.data + other.data

        assert len(self.data) == len(other.data)
        return Tensor([e1 + e2 for e1, e2 in zip(self.data, other.data)])
    
    def __mul__(self, other):
        if isinstance(self.data, int) and isinstance(other.data, int):
            return self.data * other.data


        self_res = all(isinstance(ele, list) for ele in self.data)
        other_res = all(isinstance(ele, list) for ele in other.data)
        if not self_res and not other_res and len(self.data) == len(other.data):
            return Tensor(np.array(self.data) * np.array(other.data))

        if len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
            return Tensor(np.multiply(self.data, other.data))

    def __matmul__(self, other):
        if self.shape[1] == other.shape[0]:
            return Tensor(np.matmul(self.data, other.data))

    def transpose(self):
        return np.transpose(self)

    def __sub__(self, other):
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