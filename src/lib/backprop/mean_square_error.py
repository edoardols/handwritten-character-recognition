import numpy as np

from src.lib.activation import activation
from src.lib.mapping import one_hot_encode
from src.lib.sigmoid import sigMatrix, dsigMatrix

def output(W, X, B):
    # W is a list of matrices
    # B is a list of vectors
    # Y_NN list of vectors
    Y_NN = []
    for i in range(0, len(B)):
        y_nn = np.zeros(len(B[i]))
        Y_NN.append(y_nn)

    for i in range(0, len(B)):
        # X is a single vector input
        A = np.zeros((len(W[i])), dtype=float)
        A = activation(W[i], X, B[i])
        Y_NN[i] = sigMatrix(A)
    return Y_NN

def delta_error(Y, W, X, B, Y_NN, DE, indexLayer, layer):
    # X is single example
    # W is a matrix
    # B is a vector

    # OUTPUT "for i=0"
    A = np.zeros((len(W)), dtype=float)
    A = activation(W, X, B)
    if layer == 'output':

        #X = Y_NN[i-1]
        #A = activation(W, X, B)
        y = one_hot_encode(Y)
        de = -(y - sigMatrix(A)) * dsigMatrix(A)


    if layer == 'hidden':

        #X = Y_NN[i-1]
        #A = activation(W, X, B)

        de = dsigMatrix(A) * np.dot(W, DE[i+1])

    if layer == 'input':

        #X = X
        #A = activation(W, X, B)
        de = dsigMatrix(A) * np.dot(W, DE[i+1])

    return de


def loss_function(Y, W, X, B, indexLayer, layer):
    # X is single example, Y is a scalar
    # return a vector
    e = np.zeros((1, len(W)))

    # W is a matrix
    # B is a vector

    # OUTPUT "for i=0"

    Y_NN = output(W, X, B)

    i = indexLayer

    if layer == 'output':

        X = Y_NN[i-1]
        de = delta_error(Y, W, X, B, Y_NN, DE, indexLayer, layer)

    if layer == 'hidden':

        X = Y_NN[i-1]
        de = delta_error(Y, W, X, B, Y_NN, DE, indexLayer, layer)

    if layer == 'input':

        X = X
        de = delta_error(Y, W, X, B, Y_NN, DE, indexLayer, layer)

    DE[i] = de

    de = de.reshape(-1, 1)
    X = X.reshape(-1, 1)
    X = np.transpose(X)
    e = np.dot(de, X)

    return e

def empirical_risk(Y, W, X, B, indexLayer, layer):
    # E is a vector with dim W
    E = np.zeros((1, len(W)))
    # X is the batch of examples
    # W is a matrix
    # B is a vector
    for i in range(0, len(X)):
        E = E + loss_function(Y[i], W, X[i], B, indexLayer, layer)
    return E
