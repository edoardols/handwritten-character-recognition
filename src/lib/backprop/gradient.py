import numpy as np

from src.lib.activation import activation
from src.lib.backprop.mean_square_error import empirical_risk
from src.lib.sigmoid import sigMatrix


# TODO implement mode type for gradient descent: batch, online, mini-batch
def gradient_descent_algorithm(Y, W, X, B, ETA, epochs=1):
    # W and B are list of matrix
    for e in range(0, epochs):
        for i in range(0, len(W)):
            # gradient descent
            layer_index = i
            E = empirical_risk(Y, W, X, B, layer_index)
            W[i] = W[i] - ETA * E
    return W