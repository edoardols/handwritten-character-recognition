import os

import pandas as pd
import numpy as np

from handwrittencharacter.lib.forward.gradient import gradient_descent_algorithm
from handwrittencharacter.lib.mapping import input_normalization_Matrix

def forward_training(PATH_MAIN_FILE, l, ETA, desired_epochs, learning_mode, batch_dimension):

    STEP = 100

    W = None
    B = None
    Etot = None

    # check if previous step exist

    folder_found = False

    epochs = 0

    path_to_main_folder = (PATH_MAIN_FILE + '/forward/training/' + 'F-' + learning_mode + '-l=' + str(l) + '-eta='
                           + str(ETA) + '/')

    if learning_mode == 'mini':
        path_to_main_folder = (PATH_MAIN_FILE + '/forward/training/' + 'F-' + learning_mode + '=' + str(batch_dimension)
                               + '-l=' + str(l) + '-eta=' + str(ETA) + '/')

    if os.path.exists(path_to_main_folder + 'epoch=' + str(desired_epochs)):
        return

    if os.path.exists(path_to_main_folder):
        folder_found = True
    else:
        os.makedirs(path_to_main_folder)

    # - 1 so skip the desired epochs value
    q = (desired_epochs / STEP) - 1

    while folder_found and q >= 0:
        path_to_existing_sub_folder = (path_to_main_folder + 'epoch=' + str(q * STEP) + '/')

        if os.path.exists(path_to_existing_sub_folder):
            epochs = (q * STEP)
            # Weights
            w = pd.read_csv(path_to_existing_sub_folder + 'W.csv', header=None)
            W = w.to_numpy()

            # Biases
            b = pd.read_csv(path_to_existing_sub_folder + 'B.csv', header=None)
            B = b.to_numpy()

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

    # Layout Neural Network
    np.random.seed(42)
    if W is None:
        # W is a matrix
        W = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, INPUT_DIMENSION))

    if B is None:
        # B is a vector
        B = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 1))

    # Bias learnable
    # Expand matrix W with a column B
    WB = np.insert(W, W.shape[1], np.transpose(B), axis=1)

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
        E = np.zeros(min(desired_epochs, STEP))

        for e in range(0, min(desired_epochs, STEP)):
            WB, E_epoch = gradient_descent_algorithm(D, WB, ETA, epochs + e, learning_mode, batch_dimension)
            E[e] = E_epoch

        Etot = np.append(Etot, E)

        print('Saving: Start')

        epochs = epochs + STEP

        path_to_new_sub_folder = (path_to_main_folder + 'epoch=' + str(epochs) + '/')

        if not os.path.exists(path_to_new_sub_folder):
            # Create the folder if it doesn't exist
            os.makedirs(path_to_new_sub_folder)

        weight = pd.DataFrame(WB[:, :WB.shape[1] - 1])
        weight.to_csv(path_to_new_sub_folder + 'W.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB[:, WB.shape[1] - 1:])
        bias.to_csv(path_to_new_sub_folder + 'B.csv', encoding='utf-8', header=False, index=False)

        empirical = pd.DataFrame(Etot)
        empirical.to_csv(path_to_new_sub_folder + 'E.csv', encoding='utf-8', header=False, index=False)

        print('Saving: Done')

    print('Training: Done')
