import numpy as np


def activation(W, X, B):
    # W dx0 is a matrix and X is a dx1 and B is a 0x1 vectors
    # 0 = #neurons in output
    # d = #neurons in input

    # ensure that this is a column vector
    X = X.reshape(-1, 1)

    A = np.zeros((len(W)), dtype=float)
    A = np.dot(W, X) + B
    return A
