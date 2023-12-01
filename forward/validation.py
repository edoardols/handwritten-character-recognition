import pandas as pd
import numpy as np
import math

def map_input(X):
    # input [0,255]
    # sigmoid [0,1]
    x = (0) + ((float(X) - 0) * (1 - (0))) / (255 - 0)
    return x

def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))


def dsigmoid(a):
    ds = sigmoid(a) * (1 - sigmoid(a))
    return ds


sigMatrix = np.vectorize(sigmoid)
dsigMatrix = np.vectorize(dsigmoid)

def activation(W, X, B):
    # W dxO is a matrix and X is a dx1 and B is a 0x1 vectors
    # O = #neurons in output
    # d = #neurons in input
    A = np.zeros((len(W)), dtype=float)
    A = np.dot(W, np.transpose(X)) + B
    return A

print('---------- Validation ----------')

W = pd.read_csv('weight-post-1000.csv', header=None)

B = np.full(10, -10.)


# dataset 10000
validation_set = pd.read_csv('../dataset/mnist_test.csv', header=None)

# structure of the NN
trainset_nrow = validation_set.shape[0]
trainset_ncolumn = validation_set.shape[1]

# Number of examples
l = 100
# parameters
XV_D = validation_set.iloc[:l, 1:]
#XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

#YV_D = validation_set.iloc[:, :1]
YV_D = validation_set.iloc[:l, :1]
YV = YV_D[0].to_numpy()

XV_MAP = np.full((XV.shape[0], XV.shape[1]), 0.)
for i in range(0, XV.shape[0]):
    for j in range(0, XV.shape[1]):
        XV_MAP[i][j] = map_input(XV[i][j])


# list of output NN for every label

def accuracy(Y, W, X, B):
    a = 0
    for i in range(0, len(X)):
        A = activation(W, X[i], B)
        Y_NN = sigMatrix(A)
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        if y_nn == Y[i]:
            a = a + 1
        a = (a / len(Y)) * 100
    return a


acc = accuracy(YV, W, XV_MAP, B)
print(acc)
