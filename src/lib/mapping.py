import numpy as np


def input_normalization(x):
    # https://rosettacode.org/wiki/Map_range
    # input [0,255]
    # sigmoid [0,1]
    x_mapped = 0 + ((float(x) - 0) * (1 - 0)) / (255 - 0)
    return x_mapped


def one_hot_encode(y):
    Y = np.zeros((10, 1), dtype=float)
    Y[y] = 1
    return Y


input_normalization_Matrix = np.vectorize(input_normalization)
