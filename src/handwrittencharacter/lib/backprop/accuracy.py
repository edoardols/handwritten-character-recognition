import numpy as np

from handwrittencharacter.lib.backprop.output import output


def accuracy(Y, WB0, WB1, WB2, X, validation_threshold):
    # error
    error_label = []
    error_patter = []
    error_output_nn = []

    a = 0

    for i in range(0, len(X)):
        # Get only the output
        _, _, Y_NN, _, _, _ = output(True, WB0, WB1, WB2, X[i].reshape(-1, 1))
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        if y_nn == Y[i] and Y_NN[y_nn] >= validation_threshold:
        # if y_nn == Y[i]:
            a = a + 1
        else:
            error_label.append(Y[i])
            error_patter.append(X[i][:len(X[i]) - 1])
            error_output_nn.append(y_nn)

    a = (a / len(Y)) * 100
    return int(a), error_label, error_patter, error_output_nn
