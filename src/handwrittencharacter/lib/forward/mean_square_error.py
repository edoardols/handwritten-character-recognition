import numpy as np
import torch

from handwrittencharacter.lib.activation import activation
from handwrittencharacter.lib.mapping import one_hot_encode
from handwrittencharacter.lib.sigmoid import sigMatrix, dsigMatrix


def delta_error(Y, WB, X):
    # Y is a single label
    # X is a single vector input
    # A = np.zeros((len(W)), dtype=float)
    A = activation(WB, X)
    y = one_hot_encode(Y)
    # y_nn = sigMatrix(A)
    y_nn = torch.sigmoid(A)

    #de = -(y - y_nn) * dsigMatrix(A)

    de = -(y - y_nn) * (torch.sigmoid(A) * (1 - torch.sigmoid(A)))

    return de, (y - y_nn)


def loss_function(Y, WB, X):
    # X is a single input vector
    # X transpose
    # X = X.reshape(1, -1)
    x = X.unsqueeze(0)

    de, plot_loss = delta_error(Y, WB, X)
    # de = de.reshape(-1, 1)
    de = de.view(-1, 1)
    # e = np.dot(de, X)
    e = torch.mm(de, x)

    return e, plot_loss


def empirical_risk(Y, WB, X):
    E = 0
    E_plot = 0

    if X.ndim == 1 or len(Y) == 1:
        loss, plot_loss = loss_function(Y, WB, X)
        E = E + loss
        # E_plot = E_plot + 0.5 * np.sum(plot_loss * plot_loss)
        E_plot = E_plot + 0.5 * torch.sum(plot_loss * plot_loss)
    else:
        # X is the batch of examples
        for i in range(0, len(X)):
            loss, plot_loss = loss_function(Y[i], WB, X[i])
            E = E + loss
            # E_plot = E_plot + 0.5*np.sum(plot_loss*plot_loss)
            E_plot = E_plot + 0.5 * torch.sum(plot_loss * plot_loss)

    return E, E_plot
