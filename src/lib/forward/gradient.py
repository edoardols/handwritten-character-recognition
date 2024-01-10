import numpy as np

from src.lib.forward.mean_square_error import empirical_risk
from src.lib.learning_method import learning_method


# def gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode):
def gradient_descent_algorithm(D, W, B, ETA, e, learning_mode):
    YB, XB = learning_method(D, learning_mode, 128)

    E_epoch = 0
    for i in range(0, len(XB)):
        # Create E are list of matrix
        E = empirical_risk(YB[i], W, XB[i], B)
        # gradient descent
        W = W - ETA * E

        E_epoch = E_epoch + np.sum(np.abs(E))

        print('epoch: ', e+1, 'Batch: ', i+1)

    return W, E_epoch
