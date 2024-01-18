import numpy as np

from handwrittencharacter.lib.sigmoid import sigMatrix


# Neural Network structure
# Output   : 10
# Hidden 2 : 16 + 1 bias
# Hidden 1 : 16 + 1 bias
# Input    : 28*28 + 1 bias

def output(WB2, Y1_NN):
    A2 = np.dot(WB2, Y1_NN)
    Y2_NN = sigMatrix(A2)

    return A2, Y2_NN


def hidden2(WB1, Y0_NN):
    A1 = np.dot(WB1, Y0_NN)
    Y1_NN = sigMatrix(A1)

    Y1_NN = np.insert(Y1_NN, Y1_NN.shape[0], np.transpose(1.0), axis=0)

    return A1, Y1_NN


def hidden1(WB0, X):
    A0 = np.dot(WB0, X)
    Y0_NN = sigMatrix(A0)

    Y0_NN = np.insert(Y0_NN, Y0_NN.shape[0], np.transpose(1.0), axis=0)

    return A0, Y0_NN