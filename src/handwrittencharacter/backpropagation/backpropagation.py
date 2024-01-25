import os

import pandas as pd
import numpy as np

from src.handwrittencharacter.lib.backprop.gradient import gradient_descent_algorithm
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix


def backpropagation_training(PATH_MAIN_FILE, l, ETA, desired_epochs, learning_mode, batch_dimension):

    STEP = 10

    W0 = None
    W1 = None
    W2 = None

    B0 = None
    B1 = None
    B2 = None

    Etot = None

    # check if previous step exist
    folder_found = False

    epochs = 0

    path_to_main_folder = (PATH_MAIN_FILE + '/backpropagation/training/' + 'B-' + learning_mode + '-l=' + str(l) + '-eta='
                           + str(ETA) + '/')

    if learning_mode == 'mini':
        path_to_main_folder = (PATH_MAIN_FILE + '/backpropagation/training/' + 'B-' + learning_mode + '=' + str(batch_dimension)
                               + '-l=' + str(l) + '-eta=' + str(ETA) + '/')

    if os.path.exists(path_to_main_folder + 'epoch=' + str(desired_epochs)):
        return

    if os.path.exists(path_to_main_folder):
        folder_found = True
    else:
        os.makedirs(path_to_main_folder)

    # - 1 so skip the desired epochs value
    q = (desired_epochs // STEP) - 1

    while folder_found and q >= 0:
        path_to_existing_sub_folder = (path_to_main_folder + 'epoch=' + str(q * STEP) + '/')

        if os.path.exists(path_to_existing_sub_folder):
            epochs = (q * STEP)

            # Weights
            w = pd.read_csv(path_to_existing_sub_folder + 'W0.csv', header=None)
            W0 = w.to_numpy()
            w = pd.read_csv(path_to_existing_sub_folder + 'W1.csv', header=None)
            W1 = w.to_numpy()
            w = pd.read_csv(path_to_existing_sub_folder + 'W2.csv', header=None)
            W2 = w.to_numpy()

            # Biases
            b = pd.read_csv(path_to_existing_sub_folder + 'B0.csv', header=None)
            B0 = b.to_numpy()
            b = pd.read_csv(path_to_existing_sub_folder + 'B1.csv', header=None)
            B1 = b.to_numpy()
            b = pd.read_csv(path_to_existing_sub_folder + 'B2.csv', header=None)
            B2 = b.to_numpy()

            # Empirical Risk
            e = pd.read_csv(path_to_existing_sub_folder + 'E.csv', header=None)
            Etot = e.to_numpy()
            break

        q = q - 1

    print('Loading dataset: Start')

    dataset = pd.read_csv(PATH_MAIN_FILE + '/../../dataset/mnist_train.csv', header=None)

    pattern = dataset.iloc[:l, 1:]
    label = dataset.iloc[:l, :1]
    # D = dataset.iloc[:l, :]

    X = pattern.to_numpy()
    # doing the input normalization here it is much faster because you do that just once
    X = input_normalization_Matrix(X)
    Y = label.to_numpy()

    print('Loading dataset: Done')

    print('Neural Network: Start')

    # input dim
    INPUT_DIMENSION = 28 * 28

    # output dim
    OUTPUT_DIMENSION = 10

    # region Layout Neural Network
    np.random.seed(42)
    if W0 is None or W1 is None or W2 is None:
        W0 = np.random.uniform(low=-1, high=1, size=(16, INPUT_DIMENSION))
        W1 = np.random.uniform(low=-1, high=1, size=(16, 16))
        W2 = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 16))

    if B0 is None or B1 is None or B2 is None:
        B0 = np.random.uniform(low=-1, high=1, size=(16, 1))
        B1 = np.random.uniform(low=-1, high=1, size=(16, 1))
        B2 = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 1))

    # endregion

    # Bias learnable
    # Expand matrix W with a column B
    WB0 = np.insert(W0, W0.shape[1], np.transpose(B0), axis=1)
    WB1 = np.insert(W1, W1.shape[1], np.transpose(B1), axis=1)
    WB2 = np.insert(W2, W2.shape[1], np.transpose(B2), axis=1)

    # Expand matrix X with a column of 1s
    X_hat = np.insert(X, X.shape[1], np.transpose(np.ones((X.shape[0], 1), dtype=float)), axis=1)

    # Regroup the dataset
    D = np.insert(X_hat, 0, np.transpose(Y), axis=1)
    print('Neural Network: Done')

    print('Training: Start')

    if Etot is None:
        # load past empirical risk
        Etot = []

    while epochs < desired_epochs:
        E = np.zeros(min(desired_epochs, STEP), dtype=float)

        for e in range(0, min(desired_epochs, STEP)):
            WB0, WB1, WB2, E_epoch = gradient_descent_algorithm(D, WB0, WB1, WB2, ETA, epochs + e, learning_mode, batch_dimension)
            E[e] = E_epoch

        Etot = np.append(Etot, E)

        print('Saving: Start')

        epochs = epochs + STEP

        path_to_new_sub_folder = (path_to_main_folder + 'epoch=' + str(epochs) + '/')

        if not os.path.exists(path_to_new_sub_folder):
            # Create the folder if it doesn't exist
            os.makedirs(path_to_new_sub_folder)

        # W0 and B0
        weight = pd.DataFrame(WB0[:, :WB0.shape[1] - 1])
        weight.to_csv(path_to_new_sub_folder + 'W0.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB0[:, WB0.shape[1] - 1:])
        bias.to_csv(path_to_new_sub_folder + 'B0.csv', encoding='utf-8', header=False, index=False)

        # W1 and B1
        weight = pd.DataFrame(WB1[:, :WB1.shape[1] - 1])
        weight.to_csv(path_to_new_sub_folder + 'W1.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB1[:, WB1.shape[1] - 1:])
        bias.to_csv(path_to_new_sub_folder + 'B1.csv', encoding='utf-8', header=False, index=False)

        # W2 and B2
        weight = pd.DataFrame(WB2[:, :WB2.shape[1] - 1])
        weight.to_csv(path_to_new_sub_folder + 'W2.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB2[:, WB2.shape[1] - 1:])
        bias.to_csv(path_to_new_sub_folder + 'B2.csv', encoding='utf-8', header=False, index=False)

        empirical = pd.DataFrame(Etot)
        empirical.to_csv(path_to_new_sub_folder + 'E.csv', encoding='utf-8', header=False, index=False)

        print('Saving: Done')

    print('Training: Done')
