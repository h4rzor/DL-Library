from utils import *
from tensor import Tensor 
import numpy as np 
import matplotlib.pyplot as plt 



a = Tensor([[1,2,3,4], [5,6,7,8]])
b = Tensor([[1,2,3,4], [5,6,7,8]])
#it's working for now with MxN matrices, i.e. 2D
c = Tensor([1,2])
d = Tensor([3,4])

d = randint(1, 9, (4,5))

res = ones_like(d)


res = normal((10000, ))
print(np.mean(res), np.std(res))