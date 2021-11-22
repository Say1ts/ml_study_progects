import random

import numpy as np
import random as rd


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def tanh_derivative(x):
    return 1 - (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)) ** 2


def softmax(X):
    expo = np.exp(X)
    expo_sum = expo.copy()
    for i in range(9):
        expo_sum[i] = np.sum(np.exp(X)[i])
    return expo/expo_sum


ALPHA = 0.001
np.random.seed(42)
random.seed(42)

x = np.array([[1,1,0],
             [1,0,1],
             [0,1,1],
             [0,1,0],
             [0,1,1],
             [0,0,1],
             [0,1,0],
             [1,1,1],
             [0,0,0]])
y = np.array([[1,0,0],
             [1,0,0],
             [0,1,0],
             [0,0,1],
             [0,1,0],
             [0,0,1],
             [0,0,1],
             [1,0,0],
             [0,1,0]])

# Forward prop
w1 = np.random.sample((3, 4))
b1 = rd.random()

w2 = np.random.sample((4, 3))
b2 = rd.random()

tensor_1 = np.dot(x, w1) + b1
hid = tanh(tensor_1)

tensor_2 = np.dot(hid, w2) + b2
ans = softmax(tensor_2)

e = ans - y

# Back prop
dE_dAns = ans - y

dE_dw2 = np.dot(hid.T, dE_dAns)
dE_db2 = dE_dAns

dE_dHid = np.dot(dE_dAns, w2.T) * tanh_derivative(tensor_1)
dE_dw1 = np.dot(x.T, dE_dHid)
dE_db1 = dE_dHid

# Correction
w1 -= ALPHA * dE_dw1
b1 -= ALPHA * dE_db1
w2 -= ALPHA * dE_dw2
b2 -= ALPHA * dE_db2

print(e)