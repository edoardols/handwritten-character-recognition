import numpy as np

import copy

from src.lib.activation import activation
from src.lib.mapping import one_hot_encode
from src.lib.sigmoid import sigMatrix, dsigMatrix

def output(W, X, B):
    # W is a list of matrices
    # B is a list of vectors
    # Y_NN list of vectors
    Y_NN = []

    for i in range(0, len(B)):
        y_nn = np.zeros((len(B[i]), 1))
        Y_NN.append(y_nn)

    # X is a single vector input step 0
    A = np.zeros((len(W[0]), 1), dtype=np.float32)
    A = activation(W[0], X, B[0])
    Y_NN[0] = sigMatrix(A)

    for i in range(1, len(B)):
        # X is a single vector input
        x = Y_NN[i-1]
        A = activation(W[i], x, B[i])
        Y_NN[i] = sigMatrix(A)
    return Y_NN


def loss_function(W, X, A, de):
    # W and B are for a fixe layer
    # de step i
    de = dsigMatrix(A) * np.dot(np.transpose(W), de)

    X = np.transpose(X)
    e = np.dot(de, X)

    return e, de


def empirical_risk(Y, W, X, B):
    E = copy.deepcopy(W)
    E_plot = 0

    if len(Y) == 1:  # for online mode
        for i in range(0, len(E)):
            E[i] = E[i] * 0

            # A = np.zeros((len(W)), dtype=float)
            x = X
            x = x.reshape(1, -1)
            Y_NN = output(W, x, B)

            # first step backprop

            # x is the input of the last layer (for eval dsigMatrix)
            x = Y_NN[(len(Y_NN) - 1) - 1]
            # A = activation(W, X, B)
            A = activation(W[len(W) - 1], x, B[len(W) - 1])
            y = one_hot_encode(Y)
            y_nn = sigMatrix(A)
            de = -(y - y_nn) * dsigMatrix(A)
            plot_loss = (y - y_nn)

            E_plot = E_plot + 0.5 * np.sum(plot_loss * plot_loss)

            ek = np.dot(de, np.transpose(x))

            # E[len(E) - 1] = 0
            E[len(E) - 1] = ek
            # backprop step >1
            for i in reversed(range(1, len(W) - 1)):
                # Iteration over layers

                x = Y_NN[i - 1]
                A = activation(W[i], x, B[i])

                ek, de = loss_function(W[i + 1], x, A, de)
                E[i] = E[i] + ek
            # last step backprop
            x = X
            x = x.reshape(1, -1)
            # A = activation(W, X, B)
            A = activation(W[0], x, B[0])
            de = dsigMatrix(A) * np.dot(np.transpose(W[1]), de)

            ek = np.dot(de, x)

            E[0] = E[0] + ek
    else:
        for i in range(0, len(E)):
            E[i] = E[i]*0

        for k in range(0, len(X)):
            #A = np.zeros((len(W)), dtype=float)
            x = X[k]
            x = x.reshape(1, -1)
            Y_NN = output(W, x, B)

            # first step backprop

            # x is the input of the last layer (for eval dsigMatrix)
            x = Y_NN[(len(Y_NN)-1) -1]
            # A = activation(W, X, B)
            A = activation(W[len(W)-1], x, B[len(W)-1])
            y = one_hot_encode(Y[k])
            y_nn = sigMatrix(A)
            de = -(y - y_nn) * dsigMatrix(A)

            plot_loss = (y - y_nn)

            E_plot = E_plot + 0.5*np.sum(plot_loss*plot_loss)

            ek = np.dot(de, np.transpose(x))

            # E[len(E) - 1] = 0
            E[len(E)-1] = ek
            # backprop step >1
            for i in reversed(range(1, len(W)-1)):
                # Iteration over layers

                x = Y_NN[i-1]
                A = activation(W[i], x, B[i])

                ek, de = loss_function(W[i+1], x, A, de)
                E[i] = E[i] + ek
            # last step backprop
            x = X[k]
            x = x.reshape(1, -1)
            # A = activation(W, X, B)
            A = activation(W[0], x, B[0])
            de = dsigMatrix(A) * np.dot(np.transpose(W[1]), de)

            ek = np.dot(de, x)

            E[0] = E[0] + ek

    return E, E_plot