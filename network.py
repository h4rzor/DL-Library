from tensor import Tensor
from utils import normal
import random 
from activations import relu
import numpy as np 
import torch


class Module:
    def zero_grad(self):
        for val in self.parameters():
            val.grad = 0
    
    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_inputs, activation=True):
        self.weight = [normal((n_inputs, 1)) for _ in range(n_inputs)]
        self.bias = Tensor(1)
        self.activation = activation 
    
    def __call__(self, x):
        res = []
        for wi, xi in zip(self.weight, x):
            wi = np.array(wi)
            xi = np.expand_dims(np.array(xi), 0)
            res.append(wi @ xi)
        bias = np.array(self.bias.data)
        res = np.add(res, bias).squeeze(0)
        res = Tensor(res, (), "")
        
        act = relu(res) if self.activation else act
        return act
    
    def parameters(self):
        return self.w + [self.bias]

