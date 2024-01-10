import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.lib.forward.gradient import gradient_descent_algorithm


def forward_training(l, ETA, desired_epochs, learning_mode):

    W = None
    B = None
    Epast = None

    # check if previous step exist

    folder_not_found = True
    q = desired_epochs // 5000

    global epochs, path_to_new_folder
    epochs = 0

    while folder_not_found and q > 0:
        previous_epochs = q * 5000
        path_to_previous_folder = ('forward/weight-csv/' + 'W-F-' + learning_mode + '-l=' + str(l) + '-epoch='
                                   + str(previous_epochs) + '-eta=' + str(ETA) + '/')

        if os.path.exists(path_to_previous_folder):
            folder_not_found = False

            for i in range(0, 5):
                path_to_previous_epochs = (path_to_previous_folder + 'W-F-' + learning_mode + '-l=' + str(l) + '-epoch='
                                           + str(previous_epochs - 1000 * i) + '-eta=' + str(ETA) + '/')
                if os.path.exists(path_to_previous_epochs):
                    epochs = (previous_epochs - 1000 * i)
                    # Weights
                    w = pd.read_csv(path_to_previous_epochs + 'W.csv', header=None)
                    W = w.to_numpy()

                    # Biases
                    b = pd.read_csv(path_to_previous_epochs + 'B.csv', header=None)
                    B = b.to_numpy()

                    # Empirical Risk
                    e = pd.read_csv(path_to_previous_epochs + 'E.csv', header=None)
                    Epast = e.to_numpy()
                    return
        q = q - 1

    print('Loading dataset: Start')

    dataset = pd.read_csv('../../data/mnist_train.csv', header=None)
    dataset = dataset.iloc[:l, :]

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
        B = np.full((OUTPUT_DIMENSION, 1), -10.)

    print('Neural Network: Done')

    print('Training: Start')

    if Epast is None:
        # load past empirical risk
        Epast = []

    while epochs < desired_epochs:
        E = np.zeros(min(desired_epochs, 1000), dtype=float)

        for e in range(0, min(desired_epochs, 1000)):
            W, E_epoch = gradient_descent_algorithm(dataset, W, B, ETA, epochs + e, learning_mode)
            E[e] = E_epoch

        Epast.extend(E)

        print('Saving: Start')
        q = epochs // 5000
        r = epochs % 5000
        folder_epochs = q * 5000
        if r > 0 or q == 0:
            folder_epochs = (q + 1) * 5000

        path_to_new_folder = ('forward/weight-csv/' + 'W-F-' + learning_mode + '-l=' + str(l) + '-epoch='
                              + str(folder_epochs) + '-eta=' + str(ETA) + '/')

        sub_folder_path = (path_to_new_folder + 'W-F-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs + 1000)
                           + '-eta=' + str(ETA) + '/')

        sub_folder_path = os.path.join(os.getcwd(), sub_folder_path)
        # Check if the folder already exists
        if not os.path.exists(sub_folder_path):
            # Create the folder if it doesn't exist
            os.makedirs(sub_folder_path)

        weight = pd.DataFrame(W)
        weight.to_csv(sub_folder_path + 'W.csv', encoding='utf-8', header=False, index=False)

        bias = pd.DataFrame(B)
        bias.to_csv(sub_folder_path + 'B.csv', encoding='utf-8', header=False, index=False)

        empirical = pd.DataFrame(E)
        empirical.to_csv(sub_folder_path + 'E.csv', encoding='utf-8', header=False, index=False)

        print('Saving: Done')

        epochs = epochs + 1000

    # plot
    x = np.arange(0, desired_epochs, 1)
    y = Epast
    plt.plot(x, y)

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    annotation_string = (r'$\eta$ = ' + str(ETA) + '\n'
                         + '#Patterns = ' + str(l) + '\n'
                         + 'Learning mode = ' + learning_mode + '\n')

    plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

    plt.savefig(path_to_new_folder + 'E')

    plt.show()

    print('Training: Done')
