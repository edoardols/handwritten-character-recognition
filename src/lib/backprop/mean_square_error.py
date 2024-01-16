import numpy as np

import copy

from src.lib.activation import activation
from src.lib.mapping import one_hot_encode
from src.lib.sigmoid import sigMatrix, dsigMatrix

def output(WB0, WB1, WB2, X):

    # X is a single vector input step 0
    A0 = activation(WB0, X)
    Y0_NN = sigMatrix(A0)

    Y0_NN = np.insert(Y0_NN, Y0_NN.shape[0], np.transpose(1.0), axis=0)
    #A0 = np.insert(A0, A0.shape[0], np.transpose(1.0), axis=0)

    A1 = activation(WB1, Y0_NN)
    Y1_NN = sigMatrix(A1)

    Y1_NN = np.insert(Y1_NN, Y1_NN.shape[0], np.transpose(1.0), axis=0)
    #A1 = np.insert(A1, A1.shape[0], np.transpose(1.0), axis=0)

    A2 = activation(WB2, Y1_NN)
    Y2_NN = sigMatrix(A2)
    return Y0_NN, Y1_NN, Y2_NN, A0, A1, A2


def loss_function(WB, X, A, de):
    # W and B are for a fixe layer
    # de step i

    # We have to drop the last column of WB because the bias doesn't have children and it doesn't participate
    # in the calculation of the delta error (backward step)
    de = dsigMatrix(A) * np.dot(np.transpose(WB[:, :WB.shape[1] - 1]), de)

    X = np.transpose(X)
    e = np.dot(de, X)

    return e, de


def empirical_risk(Y, WB0, WB1, WB2, X):
    E0 = copy.deepcopy(WB0 * 0)
    E1 = copy.deepcopy(WB1 * 0)
    E2 = copy.deepcopy(WB2 * 0)

    E_plot = 0
    # TODO add case for ONLINE MODE

    for k in range(0, len(X)):


        x = X[k]
        x = x.reshape(1, -1)
        Y0_NN, Y1_NN, Y2_NN, A0, A1, A2 = output(WB0, WB1, WB2, x)

        # first step backprop (output layer)

        y = one_hot_encode(Y[k])
        de = -(y - Y2_NN) * dsigMatrix(A2)
        ek = np.dot(de, np.transpose(Y1_NN))

        E2 = E2 + ek

        # second step backprop (top hidden layer)

        ek, de = loss_function(WB2, Y0_NN, A1, de)

        E1 = E1 + ek

        # second step backprop (bottom hidden layer)
        x = X[k]
        x = x.reshape(-1, 1)

        ek, de = loss_function(WB1, x, A0, de)

        E0 = E0 + ek

        plot_loss = (y - Y2_NN)
        E_plot = E_plot + 0.5 * np.sum(plot_loss * plot_loss)

    return E0, E1, E2, E_plot