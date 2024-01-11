import numpy as np

from src.lib.sigmoid import sigMatrix
from src.lib.activation import activation


def accuracy(Y, W, X, B):
    # error
    error_label = []
    error_patter = []
    error_output_nn = []

    a = 0

    for i in range(0, len(X)):
        A = activation(W, X[i], B)
        Y_NN = sigMatrix(A)
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        # if y_nn == Y[i] and Y_NN[y_nn] > 0.6:
        if y_nn == Y[i]:
            a = a + 1
        else:
            error_label.append(Y[i])
            error_patter.append(X[i])
            error_output_nn.append(y_nn)

    a = (a / len(Y)) * 100
    return int(a), error_label, error_patter, error_output_nn
