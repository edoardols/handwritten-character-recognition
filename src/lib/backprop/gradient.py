
from src.lib.backprop.mean_square_error import *
from src.lib.learning_method import learning_method


def gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode):
    # W is a list of matrices
    # B is list of vectors
    YB, XB = learning_method(Y, X, learning_mode, 128)

    E_epoch = 0
    for i in range(0, len(XB)):
        # Create E are list of matrix
        E = empirical_risk(YB[i], W, XB[i], B)

        for j in range(0, len(W)):
            W[j] = W[j] - ETA * E[j]

        E_epoch = E_epoch + E[len(E)-1].sum()

        print('epoch: ', e+1, 'Batch: ', i+1)
    return W, E_epoch