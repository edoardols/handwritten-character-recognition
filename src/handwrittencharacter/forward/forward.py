import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from handwrittencharacter.lib.forward.gradient import gradient_descent_algorithm
from handwrittencharacter.lib.mapping import input_normalization_Matrix

global epochs, path_to_new_folder

def forward_training(l, ETA, desired_epochs, learning_mode):

    STEP = 500
    SUB_STEP = 100

    W = None
    B = None
    Etot = None

    # check if previous step exist

    folder_not_found = True
    q = desired_epochs // STEP

    epochs = 0

    while folder_not_found and q >= 0:
        previous_epochs = (q + 1) * STEP
        # previous_epochs = q * STEP
        path_to_previous_folder = ('forward/training/' + 'F-' + learning_mode + '-l=' + str(l) + '-eta=' + str(ETA) +
                                   '-epoch=' + str(previous_epochs) + '/')

        if os.path.exists(path_to_previous_folder):
            folder_not_found = False

            for i in range(0, 5):
                # if previous_epochs - SUB_STEP * i <= desired_epochs:
                    path_to_previous_epochs = (path_to_previous_folder + 'epoch=' + str(previous_epochs - SUB_STEP * i)
                                               + '/')
                    if os.path.exists(path_to_previous_epochs):
                        epochs = (previous_epochs - SUB_STEP * i)
                        # Weights
                        w = pd.read_csv(path_to_previous_epochs + 'W.csv', header=None)
                        W = w.to_numpy()

                        # Biases
                        b = pd.read_csv(path_to_previous_epochs + 'B.csv', header=None)
                        B = b.to_numpy()

                        # Empirical Risk
                        e = pd.read_csv(path_to_previous_epochs + 'E.csv', header=None)
                        Etot = e.to_numpy()
                        break
        q = q - 1

    print('Loading dataset: Start')

    dataset = pd.read_csv('../../dataset/mnist_train.csv', header=None)

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
        # B = np.full((OUTPUT_DIMENSION, 1), -10.)
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
        E = np.zeros(min(desired_epochs, SUB_STEP))

        for e in range(0, min(desired_epochs, SUB_STEP)):
            WB, E_epoch = gradient_descent_algorithm(D, WB, ETA, epochs + e, learning_mode)
            E[e] = E_epoch

        Etot = np.append(Etot, E)

        print('Saving: Start')

        q = epochs // STEP
        r = epochs % STEP
        folder_epochs = q * STEP
        if r >= 0 or q == 0:
            folder_epochs = (q + 1) * STEP

        path_to_new_folder = ('forward/training/' + 'F-' + learning_mode + '-l=' + str(l) + '-eta=' + str(ETA) +
                              '-epoch=' + str(folder_epochs) + '/')

        sub_folder_path = (path_to_new_folder + 'epoch=' + str(epochs + SUB_STEP) + '/')

        sub_folder_path = os.path.join(os.getcwd(), sub_folder_path)
        # Check if the folder already exists
        if not os.path.exists(sub_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(sub_folder_path)

        weight = pd.DataFrame(WB[:, :WB.shape[1] - 1])
        weight.to_csv(sub_folder_path + 'W.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB[:, WB.shape[1] - 1:])
        bias.to_csv(sub_folder_path + 'B.csv', encoding='utf-8', header=False, index=False)

        empirical = pd.DataFrame(Etot)
        empirical.to_csv(sub_folder_path + 'E.csv', encoding='utf-8', header=False, index=False)

        print('Saving: Done')

        epochs = epochs + SUB_STEP

        if epochs // STEP:
            # plot
            x = np.arange(0, epochs, 1)
            y = Etot
            plt.plot(x, y, color='cyan')

            plt.xlabel('Epochs')
            plt.ylabel('Empirical risk')
            annotation_string = (r'$\eta$ = ' + str(ETA) + '\n'
                                 + '#Patterns = ' + str(l) + '\n'
                                 + 'Learning mode = ' + learning_mode + '\n')

            plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

            plt.savefig(path_to_new_folder + 'E')

    print('Training: Done')
