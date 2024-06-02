import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from my_layer import Affine, SoftmaxWithLoss
from my_function import softmax, cov_one_hot, cross_entropy
from time import time
np.random.seed(42)

x, t = make_circles(n_samples = 100000,noise = 0.1)
t = cov_one_hot(t)






w1 = np.random.rand(2,10)
b1 = np.random.rand(10)


w2 = np.random.rand(10,2)
b2 = np.random.rand(2)

aff1 = Affine(w1,b1)
aff2 = Affine(w2,b2)
last = SoftmaxWithLoss()


for i in range(10000):
    print(x.shape)
    hidden_x = aff1.forward(x)
    y = aff2.forward(hidden_x)
    loss = last.forward(t = t, y = y)
    dx = last.backward()
    dx = aff2.backward(dx)
    aff1.backward(dx)

    aff2.W -= 0.001 * aff2.dw
    aff2.b -= 0.001 * aff2.db

    aff1.W -= 0.001 * aff1.dw
    aff1.b -= 0.001 * aff1.db

    print(loss)
