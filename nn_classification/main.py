import random

import numpy as np
import random as rd
np.random.seed(42)
random.seed(42)


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def softmax(X):
    expo = np.exp(X)
    expo_sum = expo.copy()
    for i in range(9):
        expo_sum[i] = np.sum(np.exp(X)[i])
    return expo/expo_sum


x = np.array([[1,1,0],
             [1,0,1],
             [0,1,1],
             [0,1,0],
             [0,1,1],
             [0,0,1],
             [0,1,0],
             [1,1,1],
             [0,0,0]])


w1 = np.random.sample((3, 4))
b1 = rd.random()

w2 = np.random.sample((4, 3))
b2 = rd.random()

a = tanh(np.dot(x, w1) + b1)
b = softmax(np.dot(a, w2) + b2)


print(b)