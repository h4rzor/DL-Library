# DL-Library
**Deep learning library project**
This is the repository of the project that I'm building. I'm aiming to achieve a proficiency in the DL libraries like Torch/Tensorflow through building it from scratch. At this point I don't aim it to be fast or efficient, just to work accordingly.

**Motivation**
My main motivation is to gain the knowledge in the DL realm and be comfortable with any library of such kind.

**Project goals**
My main goal is to implement the forward and backward passes, because I think they're the hardest thing to implement from scratch, especially the backward pass with all its gradients.

**Roadmap**
I'm currently working on all things tensor related. For example tensor.randint(), exp(tensor), tanh(tensor). When I feel that it's enough for a neural network, I will start the forward and backward passes.

**Code example**
Well, it's not spectacular or anything, but:
norm_dist =  normal((3,1))
print(norm_dist)
output: Tensor([[0.22644362643370078], [1.7113118086988652], [0.0646192043883307]])

argmax =  argmax(norm_dist)
print(argmax)
output: 1
