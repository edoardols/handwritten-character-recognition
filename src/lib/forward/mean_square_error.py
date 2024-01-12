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
    y_nn = sigMatrix(A)
    de = -(y - y_nn) * dsigMatrix(A)
    return de, (y - y_nn)


def loss_function(Y, W, X, B):
    # X is a single input vector
    # X transpose
    X = X.reshape(1, -1)

    de, plot_loss = delta_error(Y, W, X, B)
    de = de.reshape(-1, 1)
    e = np.dot(de, X)
    return e, plot_loss


def empirical_risk(Y, W, X, B):
    E = 0
    E_plot = 0

    if X.ndim == 1 or len(Y) == 1:
        loss, plot_loss = loss_function(Y, W, X, B)
        E = E + loss
    else:
        # X is the batch of examples
        for i in range(0, len(X)):
            loss, plot_loss = loss_function(Y[i], W, X[i], B)
            E = E + loss
            E_plot = E_plot + 0.5*np.sum(plot_loss*plot_loss)

    return E, E_plot
