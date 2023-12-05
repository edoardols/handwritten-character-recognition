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

def delta_error(Y, W, X, B, Y_NN):
    # X is single example
    # W is a matrix
    # B is a vector

    # OUTPUT "for i=0"


    if i == 'output':
        A = np.zeros((len(W)), dtype=float)
        A = activation(W, X, B)
        y = one_hot_encode(Y)
        de = -(y - sigMatrix(A)) * dsigMatrix(A)

    if i == 'hidden':
        A = np.zeros((len(W)), dtype=float)
        A = activation(W, X, B)

    if i == 'input'


    #TODO IF layer output => "classic delta error"
    #TODO ELSE layer hidden i => de = needs de i-1
    # Y is a single label
    # X is a single vector input

    return de


def loss_function(Y, W, X, B):
    # X is a single input vector


    # OUTPUT Y fixed

    # if i == len(W)-1:
    #     x = Y_NN[i-1]
    #     E = loss_function(Y, W[i], X, B[i], i)

    # FIRST_HIDDEN X fixed
    # elif i == 0:
    #     Y = Y_NN[i]
    #     E = loss_function(Y, W[i], X, B[i])

    # HIDDEN Y and X vary


    #TODO IF layer output => "forward error"
    #TODO ELSE layer hidden i => X = "Y-1" output layer i-1

    Y_NN = output(W, X, B)

    de = delta_error(Y, W, X, B)

    de = de.reshape(-1, 1)
    X = X.reshape(-1, 1)
    X = np.transpose(X)
    e = np.dot(de, X)
    return e

def empirical_risk(Y, W, X, B):
    E = 0
    # X is the batch of examples
    # W is a list of Matrices
    # B is a list of Vectors
    for i in range(0, len(X)):
        # TODO INDEX LAYER
        E = E + loss_function(Y[i], W, X[i], B)
    return E
