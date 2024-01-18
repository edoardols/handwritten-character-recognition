import numpy as np

from src.XOR.lib.sigmoid import sigMatrix
from src.XOR.lib.activation import activation


def output(WB0, WB1, X):

    A0 = activation(WB0, X)
    Y0_NN = sigMatrix(A0)

    Y0_NN = np.insert(Y0_NN, Y0_NN.shape[0], np.transpose(1.0), axis=0)

    A1 = activation(WB1, Y0_NN)
    Y1_NN = sigMatrix(A1)

    return Y1_NN


def accuracy(Y, WB0, WB1, X):

    a = 0
    wrong_classified = 0
    unclassified = 0

    for i in range(0, len(X)):
        Y_NN = output(WB0, WB1, X[i])
        # index of the max element in the array
        # what character the NN thinks it has recognized
        if Y_NN >= 0.8:
            y_nn = 1
        elif Y_NN <= 0.2:
            y_nn = 0
        else:
            unclassified = unclassified + 1

        if y_nn == Y[i]:
            a = a + 1
        else:
            wrong_classified = wrong_classified + 1

    a = (a / len(Y)) * 100

    return int(a), int(unclassified), int(wrong_classified)
