import numpy as np
import tensorflow as tf

from handwrittencharacter.lib.activation import activation
from handwrittencharacter.lib.sigmoid import sigMatrix


def output(IS_TENSORFLOW_ENABLE, WB0, WB1, WB2, X):

    if IS_TENSORFLOW_ENABLE:
        A0 = tf.matmul(WB0, X)
        Y0_NN = tf.nn.sigmoid(A0)

        # Expand Y0_NN with 1 input for bias
        Y0_NN = tf.concat([Y0_NN, [[1.0]]], axis=0)

        A1 = tf.matmul(WB1, Y0_NN)
        Y1_NN = tf.nn.sigmoid(A1)

        # Expand Y1_NN with 1 input for bias
        Y1_NN = tf.concat([Y1_NN, [[1.0]]], axis=0)

        A2 = tf.matmul(WB2, Y1_NN)
        Y2_NN = tf.nn.sigmoid(A2)

    else:
        A0 = activation(WB0, X)
        Y0_NN = sigMatrix(A0)

        Y0_NN = np.insert(Y0_NN, Y0_NN.shape[0], np.transpose(1.0), axis=0)

        A1 = activation(WB1, Y0_NN)
        Y1_NN = sigMatrix(A1)

        Y1_NN = np.insert(Y1_NN, Y1_NN.shape[0], np.transpose(1.0), axis=0)

        A2 = activation(WB2, Y1_NN)
        Y2_NN = sigMatrix(A2)

    return Y0_NN, Y1_NN, Y2_NN, A0, A1, A2
