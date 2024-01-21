import numpy as np
import tensorflow as tf

from src.handwrittencharacter.lib.learning_method import learning_method
from src.handwrittencharacter.lib.backprop.mean_square_error import empirical_risk


def gradient_descent_algorithm(IS_TENSORFLOW_ENABLE, D, WB0, WB1, WB2, ETA, e, learning_mode, batch_dimension=128):
    global i
    # YB, XB = learning_method(IS_TENSORFLOW_ENABLE, D, learning_mode, batch_dimension)

    # shuffle the dataset
    if IS_TENSORFLOW_ENABLE:
        D = tf.random.shuffle(D)
        D = tf.transpose(D)
    else:
        np.random.shuffle(D)

    E_plot = 0

    if learning_mode == 'batch':
        X = D[:, 1:]
        Y = D[:, :1]

        E0, E1, E2, E_plot_batch = empirical_risk(IS_TENSORFLOW_ENABLE, Y, WB0, WB1, WB2, X)

        WB0 = WB0 - ETA * E0
        WB1 = WB1 - ETA * E1
        WB2 = WB2 - ETA * E2

        E_plot = E_plot + E_plot_batch

        print('epoch: ', e + 1)

    if learning_mode == 'mini':
        q = D.shape[0] // batch_dimension
        for i in range(0, q + 1):
            if i == q:
                X = D[i * batch_dimension: (i + 1) * batch_dimension, 1:]
                Y = D[i * batch_dimension: (i + 1) * batch_dimension, :1]
            else:
                # get all the remaining element
                X = D[i * batch_dimension:, 1:]
                Y = D[i * batch_dimension:, :1]

            E0, E1, E2, E_plot_batch = empirical_risk(IS_TENSORFLOW_ENABLE, Y, WB0, WB1, WB2, X)

            WB0 = WB0 - ETA * E0
            WB1 = WB1 - ETA * E1
            WB2 = WB2 - ETA * E2

            E_plot = E_plot + E_plot_batch

            print('epoch: ', e + 1, 'Batch: ', i+1)

    if learning_mode == 'online':
        ONLINE_MODE = True
        for i in range(0, int(D.shape[1])):
            # Get only one element
            # X = D[i:i+1, 1:]
            # Y = D[i:i+1, :1]

            # testing using transpose on D, so all x from row vectors became column vectors
            X = D[1:, i:i+1]
            Y = D[:1, i:i + 1]

            E0, E1, E2, E_plot_batch = empirical_risk(IS_TENSORFLOW_ENABLE, Y, WB0, WB1, WB2, X, ONLINE_MODE)

            WB0 = WB0 - ETA * E0
            WB1 = WB1 - ETA * E1
            WB2 = WB2 - ETA * E2

            E_plot = E_plot + E_plot_batch

            print('epoch: ', e+1, 'Batch: ', i+1)

    return WB0, WB1, WB2, E_plot
