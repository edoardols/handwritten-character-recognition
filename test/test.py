import numpy as np
import math

A = np.array([1,3,4,5,2,6,7,3,7,2])

B = np.dot(A,5)
#print(A*B)

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

#print(dsigMatrix(A))

a = 20

def sum():
    global a
    a = a + 10
    return 1, 2, 3

#print(a)
a, b, c = sum()
print(c)
#print(a)


v = [[1, 2], [2, 3]]
u = [[1, 2], [2, 3]]

#z = v[1] - u[1]
#print(z)


y = np.array([1, 2, 3])

x = np.array([[1, 2],[2, 3],[3, 4]])

print(y)
print(x.T)
#c = np.concatenate((x, y.T), axis=0)

print(c)

new_array = np.insert(x, 0, y, axis=1)

#view updated array

print(new_array)