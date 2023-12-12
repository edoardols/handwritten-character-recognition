
from src.lib.backprop.mean_square_error import *


# TODO implement mode type for gradient descent: batch, online, mini-batch
def gradient_descent_algorithm(Y, W, X, B, ETA, epochs=1):
    # W is a list of matrices
    # B is list of vectors
    for e in range(0, epochs):
        # Create E are list of matrix
        print('epochs: ', e)
        E = empirical_risk(Y, W, X, B)
        for i in range(0, len(W)):
            W[i] = W[i] - ETA * E[i]
    return W