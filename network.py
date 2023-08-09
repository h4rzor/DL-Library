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
        self.weight = [Tensor(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.bias = Tensor(1)
        self.activation = activation 
    
    def __call__(self, x):
        res = Tensor(0, (), "")
        for wi, xi in zip(self.weight, x):
            res += (wi.data * xi)
        act = relu(res) if self.activation else res
        return act
    
    def parameters(self):
        return self.weight + [self.bias]


class Layer(Module):
    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]



class MLP(Module):
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1], activation=i!=len(n_outputs)-1) for i in range(len(n_outputs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            #print(x)
        return x 

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]