import numpy as np
import tensorflow as tf

def activation(WB, X):
    # W dx0 is a matrix and X is a dx1 and B is a 0x1 vectors
    # 0 = #neurons in output
    # d = #neurons in input

    # ensure that this is a column vector
    # X = X.reshape(-1, 1)
    X = tf.reshape(X, (-1, 1))

    # A = np.zeros((len(W), 1), dtype=float)
    # A = np.dot(W, X) + B
    # A = np.dot(WB, X)
    A = tf.matmul(WB, X)
    return A
