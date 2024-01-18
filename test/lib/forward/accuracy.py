import numpy as np

from handwrittencharacter.lib import sigMatrix
from handwrittencharacter.lib import activation


def accuracy(Y, WB, X, validation_threshold):
    # error
    error_label = []
    error_patter = []
    error_output_nn = []

    a = 0

    for i in range(0, len(X)):
        A = activation(WB, X[i])
        Y_NN = sigMatrix(A)
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        if y_nn == Y[i] and Y_NN[y_nn] >= validation_threshold:
        # if y_nn == Y[i]:
            a = a + 1
        else:
            error_label.append(Y[i])

            # drop expansion for X
            # drop last element of X[i], 1 used for bias
            error_patter.append(X[i][:len(X[i]) - 1])
            error_output_nn.append(y_nn)

    a = (a / len(Y)) * 100
    return int(a), error_label, error_patter, error_output_nn
