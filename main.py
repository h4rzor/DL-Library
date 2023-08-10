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
res = mlp(x) 



x1 = Tensor(2.0, (), "", "x1")
x2 = Tensor(0.0, (), "", "x2")

w1 = Tensor(-3.0, (), "", "w1")
w2 = Tensor(1.0, (), "", "w2")

bias = Tensor(6.88, (), "", "bias")

x1w1 = x1 * w1
x2w2 = x2 * w2

x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + bias
o = n.tanh()
o.backward() '''




xs = Tensor([
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
])
ys = Tensor([1.0, -1.0, -1.0, 1.0]) # desired targets


mlp = MLP(3, [4,4,1])

y_pred = mlp(xs[0])

y_true = ys[0]


loss = (Tensor(y_true, (), "", "") - y_pred) ** Tensor(2, (), "","")



for n in mlp.parameters():
    n.grad = 0

loss.backward()

quit()

#x = Tensor([2.0, 3.0, -1.0])
mlp = MLP(3, [4,4,1])
#res = mlp(x) 
#print(res)
#forward pass
for _ in range(100):
    ypred = [mlp(n) for n in xs]
    
    loss = sum((y_true - y_pred.data)**2 for y_true, y_pred in zip(ys, ypred))

    for p in mlp.parameters():
        p.grad = 0.0

    for y_pred in ypred:
        y_pred.backward()

    for p in mlp.parameters():
        p.data += -0.01 * p.grad 
    


for i in range(100):
    x = [2.0, 3.0, -1.0]
    mlp = MLP(3, [4, 4, 1])
    output = mlp(x)
    target = Tensor(3.6, (), "","")

    loss = Tensor((target.data - output.data) ** 2, (target, output), "","")
    print(loss)
    for n in mlp.parameters():
        n.grad = 0.0
    loss.backward()

    for n in mlp.parameters():
        n.data += -0.0001 * n.grad 
