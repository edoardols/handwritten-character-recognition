import numpy as np

import copy

from src.XOR.lib.activation import activation
from src.XOR.lib.sigmoid import sigMatrix, dsigMatrix

def output(WB0, WB1, X):

    # X is a single vector input step 0
    A0 = activation(WB0, X)
    Y0_NN = sigMatrix(A0)

    Y0_NN = np.insert(Y0_NN, Y0_NN.shape[0], np.transpose(1.0), axis=0)

    A1 = activation(WB1, Y0_NN)
    Y1_NN = sigMatrix(A1)

    return Y0_NN, Y1_NN, A0, A1


def loss_function(WB, X, A, de):
    # W and B are for a fixe layer
    # de step i

    # We have to drop the last column of WB because the bias doesn't have children and it doesn't participate
    # in the calculation of the delta error (backward step)
    de = dsigMatrix(A) * np.dot(np.transpose(WB[:, :WB.shape[1] - 1]), de)

    X = np.transpose(X)
    e = np.dot(de, X)

    return e, de


def empirical_risk(Y, WB0, WB1, X):
    E0 = copy.deepcopy(WB0 * 0)
    E1 = copy.deepcopy(WB1 * 0)

    E_plot = 0
    # TODO add case for ONLINE MODE

    for k in range(0, len(X)):


        x = X[k]
        x = x.reshape(1, -1)
        Y0_NN, Y1_NN, A0, A1 = output(WB0, WB1, x)

        # first step backprop (output layer)

        y = Y[k]
        de = -(y - Y1_NN) * dsigMatrix(A1)
        ek = np.dot(de, np.transpose(Y0_NN))

        E1 = E1 + ek

        # second step backprop (hidden layer)

        x = X[k]
        x = x.reshape(-1, 1)

        ek, de = loss_function(WB1, x, A0, de)

        E0 = E0 + ek

        plot_loss = (y - Y1_NN)
        E_plot = E_plot + 0.5 * np.sum(plot_loss * plot_loss)

    return E0, E1, E_plot