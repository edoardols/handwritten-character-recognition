import numpy as np
import math


def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))


def dsigmoid(a):
    ds = sigmoid(a) * (1 - sigmoid(a))
    return ds


sigMatrix = np.vectorize(sigmoid)
dsigMatrix = np.vectorize(dsigmoid)
