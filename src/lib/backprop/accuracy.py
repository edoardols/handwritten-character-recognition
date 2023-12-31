import numpy as np

from src.lib.sigmoid import sigMatrix
from src.lib.activation import activation

def output(W, X, B):
    # W is a list of matrices
    # B is a list of vectors
    # Y_NN list of vectors
    Y_NN = []

    for i in range(0, len(B)):
        y_nn = np.zeros((len(B[i]), 1))
        Y_NN.append(y_nn)

    # X is a single vector input step 0
    A = np.zeros((len(W[0]), 1), dtype=float)
    A = activation(W[0], X, B[0])
    Y_NN[0] = sigMatrix(A)

    for i in range(1, len(B)):
        # X is a single vector input
        x = Y_NN[i-1]
        A = activation(W[i], x, B[i])
        Y_NN[i] = sigMatrix(A)
    return Y_NN[len(Y_NN)-1]

def accuracy(Y, W, X, B):
    a = 0
    for i in range(0, len(X)):
        Y_NN = output(W, X[i], B)
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        if y_nn == Y[i]:
            a = a + 1
    a = (a / len(Y)) * 100
    return int(a)