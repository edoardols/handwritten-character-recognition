import numpy as np
import tensorflow as tf

from handwrittencharacter.lib.mapping import one_hot_encode
from handwrittencharacter.lib.sigmoid import dsigMatrix
from handwrittencharacter.lib.backprop.output import output


def loss_function(IS_TENSORFLOW_ENABLE, WB, X, A, de):

    # We have to drop the last column of WB because the bias doesn't have children, and it doesn't participate
    # in the calculation of the delta error (backward step)
    if IS_TENSORFLOW_ENABLE:
        de = tf.nn.sigmoid(A) * tf.matmul(tf.transpose(WB[:, :WB.shape[1] - 1]), de)
        X = tf.transpose(X)
        e = tf.matmul(de, X)
        # Alternatively, you can use the @ operator
        # e = de @ X
    else:
        de = dsigMatrix(A) * np.dot(np.transpose(WB[:, :WB.shape[1] - 1]), de)
        X = np.transpose(X)
        e = np.dot(de, X)

    return e, de


def empirical_risk(IS_TENSORFLOW_ENABLE, Y, WB0, WB1, WB2, X, ONLINE_MODE=False):

    if IS_TENSORFLOW_ENABLE:
        E0 = tf.zeros((16, 785), dtype=tf.float64)
        E1 = tf.zeros((16, 17), dtype=tf.float64)
        E2 = tf.zeros((10, 17), dtype=tf.float64)

    else:
        E0 = np.zeros((16, 785), dtype=float)
        E1 = np.zeros((16, 17), dtype=float)
        E2 = np.zeros((10, 17), dtype=float)

    E_plot = 0

    for k in range(0, int(X.shape[1])):

        if not ONLINE_MODE:
            x = X[k]
            x = tf.reshape(x, (-1, 1))

            y = one_hot_encode(Y[k])
            y = tf.reshape(y, (-1, 1))
        else:
            x = X

            y = one_hot_encode(Y)
            y = tf.reshape(y, (-1, 1))

        Y0_NN, Y1_NN, Y2_NN, A0, A1, A2 = output(IS_TENSORFLOW_ENABLE, WB0, WB1, WB2, x)

        # first step backprop (output layer)

        plot_loss = (y - Y2_NN)

        if IS_TENSORFLOW_ENABLE:
            # Original
            # de = - (y - Y2_NN) * (tf.nn.sigmoid(A2) * (1 - tf.nn.sigmoid(A2)))

            # Fast
            # de = - (y - Y2_NN) * (Y2_NN * (1.0 - Y2_NN))

            # Fastest
            de = - plot_loss * (Y2_NN * (1.0 - Y2_NN))
        else:
            # Original
            # de = -(y - Y2_NN) * dsigMatrix(A2)

            # Fast
            # de = - (y - Y2_NN) * (Y2_NN * (1.0 - Y2_NN))

            # Fastest
            de = - plot_loss * (Y2_NN * (1.0 - Y2_NN))

        if IS_TENSORFLOW_ENABLE:
            ek = tf.matmul(de, tf.transpose(Y1_NN))
        else:
            ek = np.dot(de, np.transpose(Y1_NN))

        E2 = E2 + ek

        # second step backprop (top hidden layer)
        ek, de = loss_function(IS_TENSORFLOW_ENABLE, WB2, Y0_NN, A1, de)
        E1 = E1 + ek

        # second step backprop (bottom hidden layer)
        ek, de = loss_function(IS_TENSORFLOW_ENABLE, WB1, x, A0, de)
        E0 = E0 + ek

        if IS_TENSORFLOW_ENABLE:
            # E_plot = E_plot + 0.5 * tf.reduce_sum(plot_loss * plot_loss)
            E_plot = E_plot + 0.5 * tf.reduce_sum(tf.square(plot_loss))
        else:
            E_plot = E_plot + 0.5 * np.sum(plot_loss * plot_loss)

    return E0, E1, E2, E_plot
