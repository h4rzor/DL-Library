from utils import *
from tensor import Tensor 
import numpy as np 
import matplotlib.pyplot as plt 
from activations import *
from network import Neuron, Layer, MLP
from graphviz import Digraph
import torch



'''a = Tensor([[1,2,3,4], [5,6,7,8]])
b = Tensor([[1,2,3,4], [5,6,7,8]])
#it's working for now with MxN matrices, i.e. 2D
c = Tensor([1,2,3,2,1,32,4,5,2,4])
d = Tensor([4,4,8,3,6,3])

d = Tensor([1,2,3,0,-1,-4,6])
tensor = randint(1,6,(4,1))

tensor = Tensor([[1,6,12]
                ,[23,2,20],
                 [21,5,3]])


a = Tensor(6)
b = Tensor(5)

res = a + b
print(res._op)
prev = list(res._prev)
print(prev[0])
print(prev[1])'''


'''neuron = Neuron(3)
res = neuron(x)
layer = Layer(7, 5)
result = layer(x)
x = Tensor([1,2,3])
nouts = [4, 4, 1]
mlp = MLP(3, nouts)
res = mlp(x) '''

a = Tensor(2.0, (), "", "a")
b = Tensor(-3.0, (), "", "b")
c = Tensor(10.0, (), "", "c")
e = a*b; e.label = "e"
d = e + c; d.label = "d"
f = Tensor(-2.0, (), "", "f")
L = d * f; L.label = "L"

L.backward()

print(f.grad, d.data) 
print(d.grad, f.data) 

print(e.grad, d.grad * 1) 
print(c.grad, d.grad * 1) 
print(a.grad, b.data * e.grad) 
print(b.grad, a.data * e.grad) 