import pandas as pd
import numpy as np

import os

from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.backprop.gradient import gradient_descent_algorithm
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix


def backpropagation_training(l, ETA, desired_epochs, learning_mode):

    STEP = 500
    SUB_STEP = 100

    W0 = None
    W1 = None
    W2 = None

    B0 = None
    B1 = None
    B2 = None

    Etot = None

    # check if previous step exist

    folder_not_found = True
    q = desired_epochs // STEP

    global epochs, path_to_new_folder
    epochs = 0

    while folder_not_found and q >= 0:
        previous_epochs = (q + 1) * STEP
        path_to_previous_folder = ('backpropagation/training/' + 'B-' + learning_mode + '-l=' + str(l) + '-eta='
                                   + str(ETA) + '-epoch=' + str(previous_epochs) + '/')

        if os.path.exists(path_to_previous_folder):
            folder_not_found = False

            for i in range(0, 5):
                path_to_previous_epochs = (path_to_previous_folder + 'epoch=' + str(previous_epochs - SUB_STEP * i)
                                           + '/')
                if os.path.exists(path_to_previous_epochs):
                    epochs = (previous_epochs - SUB_STEP * i)

                    # Weights
                    w = pd.read_csv(path_to_previous_epochs + 'W0.csv', header=None)
                    W0 = w.to_numpy()
                    w = pd.read_csv(path_to_previous_epochs + 'W1.csv', header=None)
                    W1 = w.to_numpy()
                    w = pd.read_csv(path_to_previous_epochs + 'W2.csv', header=None)
                    W2 = w.to_numpy()

                    # Biases
                    b = pd.read_csv(path_to_previous_epochs + 'B0.csv', header=None)
                    B0 = b.to_numpy()
                    b = pd.read_csv(path_to_previous_epochs + 'B1.csv', header=None)
                    B1 = b.to_numpy()
                    b = pd.read_csv(path_to_previous_epochs + 'B2.csv', header=None)
                    B2 = b.to_numpy()

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
        E = np.zeros(min(desired_epochs, SUB_STEP), dtype=float)

        for e in range(0, min(desired_epochs, SUB_STEP)):
            WB0, WB1, WB2, E_epoch = gradient_descent_algorithm(D, WB0, WB1, WB2, ETA, epochs + e, learning_mode)
            E[e] = E_epoch

        Etot = np.append(Etot, E)

        print('Saving: Start')
        q = epochs // STEP
        r = epochs % STEP
        folder_epochs = q * STEP
        if r >= 0 or q == 0:
            folder_epochs = (q + 1) * STEP

        path_to_new_folder = ('backpropagation/training/' + 'B-' + learning_mode + '-l=' + str(l) + '-eta=' + str(ETA) +
                              '-epoch=' + str(folder_epochs) + '/')

        sub_folder_path = (path_to_new_folder + 'epoch=' + str(epochs + SUB_STEP) + '/')

        sub_folder_path = os.path.join(os.getcwd(), sub_folder_path)
        # Check if the folder already exists
        if not os.path.exists(sub_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(sub_folder_path)

        # W0 and B0
        weight = pd.DataFrame(WB0[:, :WB0.shape[1] - 1])
        weight.to_csv(sub_folder_path + 'W0.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB0[:, WB0.shape[1] - 1:])
        bias.to_csv(sub_folder_path + 'B0.csv', encoding='utf-8', header=False, index=False)

        # W1 and B1
        weight = pd.DataFrame(WB1[:, :WB1.shape[1] - 1])
        weight.to_csv(sub_folder_path + 'W1.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB1[:, WB1.shape[1] - 1:])
        bias.to_csv(sub_folder_path + 'B1.csv', encoding='utf-8', header=False, index=False)

        # W2 and B2
        weight = pd.DataFrame(WB2[:, :WB2.shape[1] - 1])
        weight.to_csv(sub_folder_path + 'W2.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(WB2[:, WB2.shape[1] - 1:])
        bias.to_csv(sub_folder_path + 'B2.csv', encoding='utf-8', header=False, index=False)

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