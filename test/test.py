import numpy as np
import math

A = np.array([1,3,4,5,2,6,7,3,7,2])

B = np.dot(A,5)
print(A*B)

def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))

def dsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

sigMatrix = np.vectorize(sigmoid)
dsigMatrix = np.vectorize(dsigmoid)

print(dsigMatrix(A))

a = 20

def sum():
    global a
    a = a + 10

print(a)
sum()
print(a)