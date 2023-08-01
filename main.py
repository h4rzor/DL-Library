from utils import *
from tensor import Tensor 
import numpy as np 
import matplotlib.pyplot as plt 
from activations import *



a = Tensor([[1,2,3,4], [5,6,7,8]])
b = Tensor([[1,2,3,4], [5,6,7,8]])
#it's working for now with MxN matrices, i.e. 2D
c = Tensor([1,2,3,2,1,32,4,5,2,4])
d = Tensor([4,4,8,3,6,3])

d = Tensor([1,2,3,0,-1,-4,6])


res = relu(d)
#Activation kinda works only for 1D and 2D for now
print(res)