import os

import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.backprop.gradient import gradient_descent_algorithm
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix

global epochs, IS_TENSORFLOW_ENABLE


def initialize_weights_and_biases():

    INPUT_DIMENSION = 28 * 28
    OUTPUT_DIMENSION = 10

    if IS_TENSORFLOW_ENABLE:
        tf.random.set_seed(42)

        W0 = tf.random.uniform(shape=(16, INPUT_DIMENSION), minval=-1, maxval=1, dtype=tf.float64)
        W1 = tf.random.uniform(shape=(16, 16), minval=-1, maxval=1, dtype=tf.float64)
        W2 = tf.random.uniform(shape=(OUTPUT_DIMENSION, 16), minval=-1, maxval=1, dtype=tf.float64)

        B0 = tf.random.uniform(shape=(16, 1), minval=-1, maxval=1, dtype=tf.float64)
        B1 = tf.random.uniform(shape=(16, 1), minval=-1, maxval=1, dtype=tf.float64)
        B2 = tf.random.uniform(shape=(OUTPUT_DIMENSION, 1), minval=-1, maxval=1, dtype=tf.float64)
    else:
        np.random.seed(42)

        W0 = np.random.uniform(low=-1, high=1, size=(16, INPUT_DIMENSION))
        W1 = np.random.uniform(low=-1, high=1, size=(16, 16))
        W2 = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 16))

        B0 = np.random.uniform(low=-1, high=1, size=(16, 1))
        B1 = np.random.uniform(low=-1, high=1, size=(16, 1))
        B2 = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 1))

    return W0, W1, W2, B0, B1, B2


def backpropagation_training(PATH_MAIN_FILE, l, ETA, desired_epochs, learning_mode, batch_dimension):

    global epochs, IS_TENSORFLOW_ENABLE

    # Initialization
    W0 = None
    W1 = None
    W2 = None

    B0 = None
    B1 = None
    B2 = None

    E_total = None

    IS_TENSORFLOW_ENABLE = False

    STEP = 5
    SUB_STEP = 1

    if tf.test.gpu_device_name():
        print("GPU found, use TensorFlow")
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        IS_TENSORFLOW_ENABLE = True
    else:
        print("GPU not found, use Numpy")

    print('Loading dataset: Start')

    dataset = pd.read_csv(PATH_MAIN_FILE + '/../../dataset/mnist_train.csv', header=None)

    pattern = dataset.iloc[:l, 1:]
    label = dataset.iloc[:l, :1]

    if IS_TENSORFLOW_ENABLE:
        X = tf.constant(pattern.values, dtype=tf.float64)
        X = X / 255.0
        Y = tf.constant(label.values, dtype=tf.float64)

        # Expand matrix X with a column of 1s
        X_hat = tf.concat([X, tf.ones((X.shape[0], 1), dtype=tf.float64)], axis=1)

        # Regroup the dataset
        D = tf.concat([Y, X_hat], axis=1)
    else:
        X = pattern.to_numpy()
        X = input_normalization_Matrix(X)
        Y = label.to_numpy()

        # Expand matrix X with a column of 1s
        X_hat = np.insert(X, X.shape[1], np.transpose(np.ones((X.shape[0], 1), dtype=float)), axis=1)

        # Regroup the dataset
        D = np.insert(X_hat, 0, np.transpose(Y), axis=1)

    print('Loading dataset: Done')

    print('Neural Network: Start')

    # check if previous step exist
    folder_not_found = True
    q = desired_epochs // STEP

    while folder_not_found and q >= 0:

        previous_epochs = (q + 1) * STEP

        path_to_previous_folder = (PATH_MAIN_FILE + '/backpropagation/training/' + 'B-' + learning_mode)

        if learning_mode == 'mini':
            path_to_previous_folder = path_to_previous_folder + '=' + str(batch_dimension)

        path_to_previous_folder = (path_to_previous_folder + '-l=' + str(l) + '-eta=' + str(ETA) + '-epoch='
                                   + str(previous_epochs) + '/')

        if os.path.exists(path_to_previous_folder):
            folder_not_found = False

            for i in range(0, 5):
                path_to_previous_epochs = (path_to_previous_folder + 'epoch=' + str(previous_epochs - SUB_STEP * i) + '/')
                if os.path.exists(path_to_previous_epochs):
                    epochs = (previous_epochs - SUB_STEP * i)

                    # Weights
                    w0 = pd.read_csv(path_to_previous_epochs + 'W0.csv', header=None)
                    w1 = pd.read_csv(path_to_previous_epochs + 'W1.csv', header=None)
                    w2 = pd.read_csv(path_to_previous_epochs + 'W2.csv', header=None)

                    # Biases
                    b0 = pd.read_csv(path_to_previous_epochs + 'B0.csv', header=None)
                    b1 = pd.read_csv(path_to_previous_epochs + 'B1.csv', header=None)
                    b2 = pd.read_csv(path_to_previous_epochs + 'B2.csv', header=None)

                    if IS_TENSORFLOW_ENABLE:
                        W0 = tf.constant(w0.values, dtype=tf.float64)
                        W1 = tf.constant(w1.values, dtype=tf.float64)
                        W2 = tf.constant(w2.values, dtype=tf.float64)

                        B0 = tf.constant(b0.values, dtype=tf.float64)
                        B1 = tf.constant(b1.values, dtype=tf.float64)
                        B2 = tf.constant(b2.values, dtype=tf.float64)
                    else:
                        W0 = w0.to_numpy()
                        W1 = w1.to_numpy()
                        W2 = w2.to_numpy()

                        B0 = b0.to_numpy()
                        B1 = b1.to_numpy()
                        B2 = b2.to_numpy()

                    # Empirical Risk
                    e = pd.read_csv(path_to_previous_epochs + 'E.csv', header=None)
                    E_total = e.to_numpy()
                    break
        q = q - 1

    # No previous weight or biases are found
    if W0 is None or W1 is None or W2 is None or B0 is None or B1 is None or B2 is None:
        W0, W1, W2, B0, B1, B2 = initialize_weights_and_biases()

    if IS_TENSORFLOW_ENABLE:
        # Expand matrix W with a column B
        WB0 = tf.concat([W0, B0], axis=1)
        WB1 = tf.concat([W1, B1], axis=1)
        WB2 = tf.concat([W2, B2], axis=1)

    else:
        # Expand matrix W with a column B
        WB0 = np.insert(W0, W0.shape[1], np.transpose(B0), axis=1)
        WB1 = np.insert(W1, W1.shape[1], np.transpose(B1), axis=1)
        WB2 = np.insert(W2, W2.shape[1], np.transpose(B2), axis=1)

    print('Neural Network: Done')

    print('Training: Start')

    if E_total is None:
        # load past empirical risk
        E_total = []

    while epochs < desired_epochs:
        E = np.zeros(min(desired_epochs, SUB_STEP), dtype=float)

        for e in range(0, min(desired_epochs, SUB_STEP)):
            WB0, WB1, WB2, E_epoch = gradient_descent_algorithm(IS_TENSORFLOW_ENABLE, D, WB0, WB1, WB2, ETA, epochs + e, learning_mode, batch_dimension)
            E[e] = E_epoch

        E_total = np.append(E_total, E)

        print('Saving: Start')

        q = epochs // STEP
        r = epochs % STEP
        folder_epochs = q * STEP
        if r >= 0 or q == 0:
            folder_epochs = (q + 1) * STEP

        path_to_new_folder = (PATH_MAIN_FILE + '/backpropagation/training/' + 'B-' + learning_mode)

        if learning_mode == 'mini':
            path_to_new_folder = path_to_new_folder + '=' + str(batch_dimension)

        path_to_new_folder = (path_to_new_folder + '-l=' + str(l) + '-eta=' + str(ETA) + '-epoch=' + str(folder_epochs) + '/')

        sub_folder_path = (path_to_new_folder + 'epoch=' + str(epochs + SUB_STEP) + '/')

        # Check if the folder already exists
        if not os.path.exists(sub_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(sub_folder_path)

        w0 = pd.DataFrame(WB0[:, :WB0.shape[1] - 1])
        w0.to_csv(sub_folder_path + 'W0.csv', encoding='utf-8', header=False, index=False)

        w1 = pd.DataFrame(WB1[:, :WB1.shape[1] - 1])
        w1.to_csv(sub_folder_path + 'W1.csv', encoding='utf-8', header=False, index=False)

        w2 = pd.DataFrame(WB2[:, :WB2.shape[1] - 1])
        w2.to_csv(sub_folder_path + 'W2.csv', encoding='utf-8', header=False, index=False)

        b0 = pd.DataFrame(WB0[:, WB0.shape[1] - 1:])
        b0.to_csv(sub_folder_path + 'B0.csv', encoding='utf-8', header=False, index=False)

        b1 = pd.DataFrame(WB1[:, WB1.shape[1] - 1:])
        b1.to_csv(sub_folder_path + 'B1.csv', encoding='utf-8', header=False, index=False)

        b2 = pd.DataFrame(WB2[:, WB2.shape[1] - 1:])
        b2.to_csv(sub_folder_path + 'B2.csv', encoding='utf-8', header=False, index=False)

        empirical = pd.DataFrame(E_total)
        empirical.to_csv(sub_folder_path + 'E.csv', encoding='utf-8', header=False, index=False)

        print('Saving: Done')

        epochs = epochs + SUB_STEP

        if epochs // STEP:
            # plot
            x = np.arange(0, epochs, 1)
            y = E_total
            plt.plot(x, y, color='cyan')

            plt.xlabel('Epochs')
            plt.ylabel('Empirical risk')
            annotation_string = (r'$\eta$ = ' + str(ETA) + '\n'
                                 + '#Patterns = ' + str(l) + '\n'
                                 + 'Learning mode = ' + learning_mode + '\n')

            plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

            plt.savefig(path_to_new_folder + 'E')

    print('Training: Done')