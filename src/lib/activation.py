import numpy as np


def activation(W, X, B):
    # W dx0 is a matrix and X is a dx1 and B is a 0x1 vectors
    # 0 = #neurons in output
    # d = #neurons in input
    A = np.zeros((len(W)), dtype=float)
    A = np.dot(W, np.transpose(X)) + B
    return A
