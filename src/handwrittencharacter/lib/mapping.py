import numpy as np
import tensorflow as tf


def input_normalization(x):
    # https://rosettacode.org/wiki/Map_range
    # input [0,255]
    # sigmoid [0,1]
    x_mapped = 0 + ((x - 0) * (1 - 0)) / (255 - 0)
    return x_mapped


def one_hot_encode(y):
    # Y = np.zeros((10, 1), dtype=float)
    Y = tf.zeros((10, 1), dtype=tf.float64)
    # Y[int(y)] = 1
    Y = tf.one_hot(int(y), 10, on_value=1.0, off_value=0.0, dtype=tf.float64)
    return Y


input_normalization_Matrix = np.vectorize(input_normalization)
