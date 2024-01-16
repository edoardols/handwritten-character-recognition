import pandas as pd
import numpy as np

import os

from matplotlib import pyplot as plt

from src.lib.backprop.gradient import gradient_descent_algorithm
from src.lib.mapping import input_normalization_Matrix


def backpropagation_training(l, ETA, desired_epochs, learning_mode):

    #STEP = 500
    #SUB_STEP = 100
    #ONLY FOR ONLINE MODE TO SAVE MORE FOLDERS
    STEP = 50
    SUB_STEP = 10

    W = []
    B = []
    Etot = None

    # configuration file
    # HLNN have structure [num_layer][num_neuron]
    HLNN = pd.read_csv('backpropagation/HLNN.csv')

    # check if previous step exist

    folder_not_found = True
    q = desired_epochs // STEP

    global epochs, path_to_new_folder
    epochs = 0

    while folder_not_found and q > 0:
        previous_epochs = q * STEP
        path_to_previous_folder = ('backpropagation/weight-csv/' + 'W-B-' + learning_mode + '-l=' + str(l) + '-epoch='
                                   + str(previous_epochs) + '-eta=' + str(ETA) + '/')

        if os.path.exists(path_to_previous_folder):
            folder_not_found = False

            for i in range(0, 5):
                path_to_previous_epochs = (path_to_previous_folder + 'W-B-' + learning_mode + '-l=' + str(l) + '-epoch='
                                           + str(previous_epochs - SUB_STEP * i) + '-eta=' + str(ETA) + '/')
                if os.path.exists(path_to_previous_epochs):
                    epochs = (previous_epochs - SUB_STEP * i)
                    for i in range(0, 3):
                        # Biases
                        b = pd.read_csv(path_to_previous_epochs + 'B' + str(i) + '.csv', header=None)
                        b = b[0].to_numpy()
                        B.append(b.reshape(-1, 1))

                        # Weights
                        w = pd.read_csv(path_to_previous_epochs + 'W' + str(i) + '.csv', header=None)
                        W.append(w.to_numpy())

                    # Empirical Risk
                    e = pd.read_csv(path_to_previous_epochs + 'E.csv', header=None)
                    Etot = e.to_numpy()
                    break
        q = q - 1

    print('Loading dataset: Start')

    dataset = pd.read_csv('../../dataset/mnist_train.csv', header=None)
    dataset = dataset.iloc[:l, :]

    D = dataset.to_numpy()

    print('Loading dataset: Done')

    print('Neural Network: Start')

    # input dim
    INPUT_DIMENSION = 28 * 28

    # output dim
    OUTPUT_DIMENSION = 10

    #region Layout Neural Network
    np.random.seed(42)
    if len(W) == 0:
        # W is a list
        for i in range(HLNN.shape[0]):
            if i == 0:
                w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[0, 1], INPUT_DIMENSION))
            else:
                w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[i, 1], HLNN.iloc[i - 1, 1]))
            W.append(w)

        w = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, HLNN.iloc[HLNN.shape[0] - 1, 1]))
        W.append(w)

    if len(B) == 0:
        # B is a list
        for i in range(HLNN.shape[0]):
            b = np.full((HLNN.iloc[i, 1], 1), 0.)
            B.append(b)

        b = np.full((OUTPUT_DIMENSION, 1), 0.)
        B.append(b)
    #endregion
    print('Neural Network: Done')

    print('Training: Start')

    if Etot is None:
        # load past empirical risk
        Etot = []

    while epochs < desired_epochs:
        E = np.zeros(min(desired_epochs, SUB_STEP), dtype=float)

        for e in range(0, min(desired_epochs, SUB_STEP)):
            W, E_epoch = gradient_descent_algorithm(D, W, B, ETA, epochs + e, learning_mode)
            E[e] = E_epoch

        Etot = np.append(Etot, E)

        print('Saving: Start')
        q = epochs // STEP
        r = epochs % STEP
        folder_epochs = q * STEP
        if r >= 0 or q == 0:
            folder_epochs = (q + 1) * STEP

        path_to_new_folder = ('backpropagation/weight-csv/' + 'W-B-' + learning_mode + '-l=' + str(l) + '-epoch='
                              + str(folder_epochs) + '-eta=' + str(ETA) + '/')

        sub_folder_path = (path_to_new_folder + 'W-B-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs + SUB_STEP)
                           + '-eta=' + str(ETA) + '/')

        sub_folder_path = os.path.join(os.getcwd(), sub_folder_path)
        # Check if the folder already exists
        if not os.path.exists(sub_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(sub_folder_path)

        for i in range(0, len(W)):
            # 0 is the input layer
            weight = pd.DataFrame(W[i])
            weight.to_csv(sub_folder_path + 'W' + str(i) + '.csv', encoding='utf-8', header=False, index=False)

        for i in range(0, len(B)):
            # 0 is the input layer
            bias = pd.DataFrame(B[i])
            bias.to_csv(sub_folder_path + 'B' + str(i) + '.csv', encoding='utf-8', header=False, index=False)

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

    # region plot
    x = np.arange(0, desired_epochs, 1)
    y = Etot
    plt.plot(x, y, color='cyan')

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    annotation_string = (r'$\eta$ = ' + str(ETA) + '\n'
                         + '#Patterns = ' + str(l) + '\n'
                         + 'Learning mode = ' + learning_mode + '\n')

    plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

    plt.savefig(path_to_new_folder + 'E')

    plt.show()

    # endregion
    print('Training: Done')