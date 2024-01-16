import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.backprop.accuracy import accuracy as backprop_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix

global current_index
current_index = 0


def XOR_validation_graph(validation_dataset_path, learning_mode, pattern, epochs, eta, validation_threshold):
    print('Validation: Start')
    validation_dataset = pd.read_csv('../../dataset/' + validation_dataset_path, header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

    accuracy = np.zeros(epochs // 100)
    for i in range(500, epochs + 500, 500):
        parent_folder = 'B-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta) + '-epoch=' + str(i)
        for j in range(i - 400, i + 100, 100):
            if j > epochs:
                break
            child_folder = 'epoch=' + str(j)
            weight_and_biases_path = parent_folder + '/' + child_folder

            path = 'backpropagation/training/' + weight_and_biases_path + '/'

            # Weights
            w = pd.read_csv(path + 'W0.csv', header=None)
            W0 = w.to_numpy()
            w = pd.read_csv(path + 'W1.csv', header=None)
            W1 = w.to_numpy()
            w = pd.read_csv(path + 'W2.csv', header=None)
            W2 = w.to_numpy()

            # Biases
            b = pd.read_csv(path + 'B0.csv', header=None)
            B0 = b.to_numpy()
            b = pd.read_csv(path + 'B1.csv', header=None)
            B1 = b.to_numpy()
            b = pd.read_csv(path + 'B2.csv', header=None)
            B2 = b.to_numpy()

            # Expand matrix W with a column B
            WB0 = np.insert(W0, W0.shape[1], np.transpose(B0), axis=1)
            WB1 = np.insert(W1, W1.shape[1], np.transpose(B1), axis=1)
            WB2 = np.insert(W2, W2.shape[1], np.transpose(B2), axis=1)

            percentage, error_label, images, error_output_nn = backprop_accuracy(YV, WB0, WB1, WB2, XV, validation_threshold)

            accuracy[(j // 100) - 1] = percentage

            print('Validation: Done')

    # plot
    x = np.arange(100, epochs + 100, 100)
    y = accuracy
    plt.plot(x, y, color='cyan')

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    plt.title('Threshold: ' + str(int(validation_threshold * 100)) + '%')
    annotation_string = (r'$\eta$ = ' + str(eta) + '\n'
                         + '#Patterns = ' + str(pattern) + '\n'
                         + 'Learning mode = ' + learning_mode + '\n')

    plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

    path_figure_folder = 'validation/figure-backpropagation/' + validation_dataset + '/threshold=' + str(
        validation_threshold) + '/W-B-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(
        epochs) + '-eta=' + str(eta)

    if not os.path.exists('validation/figure-backpropagation/threshold=' + str(validation_threshold)):
        # Create the folder if it doesn't exist
        os.makedirs('validation/figure-backpropagation/threshold=' + str(validation_threshold))
    plt.savefig(path_figure_folder + '.png')

    plt.show()
