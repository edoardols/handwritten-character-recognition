import numpy as np

from src.lib.sigmoid import sigMatrix
from src.lib.activation import activation


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