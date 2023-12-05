import numpy as np

from src.lib.activation import activation
from src.lib.mapping import one_hot_encode
from src.lib.sigmoid import sigMatrix, dsigMatrix


def delta_error(Y, W, X, B):
    # Y is a single label
    # X is a single vector input
    A = np.zeros((len(W)), dtype=float)
    A = activation(W, X, B)
    y = one_hot_encode(Y)
    de = -(y - sigMatrix(A)) * dsigMatrix(A)
    return de


def empirical_risk(Y, W, X, B):
    # X is a single input vector
    de = delta_error(Y, W, X, B)
    de = de.reshape(-1, 1)
    X = X.reshape(-1, 1)
    X = np.transpose(X)
    e = np.dot(de, X)
    return e


def loss_function(Y, W, X, B):
    E = 0
    # X is the batch of examples
    for i in range(0, len(X)):
        E = E + empirical_risk(Y[i], W, X[i], B)
    return E
