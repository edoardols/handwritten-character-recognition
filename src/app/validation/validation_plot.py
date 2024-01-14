import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.lib.forward.accuracy import accuracy as forward_accuracy
from src.lib.backprop.accuracy import accuracy as backprop_accuracy
from src.lib.mapping import input_normalization_Matrix

global current_index
current_index = 0


def forward_validation_graph(validation_dataset_name, weight_and_biases_path, validation_threshold):
    print('Validation: Start')

    folders = weight_and_biases_path.split('/')

    parameters = folders[0].split('-')

    pattern = parameters[2].split('=')[1]
    eta = parameters[3].split('=')[1]
    learning_mode = parameters[1]
    epochs = int(folders[1].split('=')[1])

    validation_dataset_path = '../../dataset/' + validation_dataset_name + '.csv'
    validation_dataset = pd.read_csv(validation_dataset_path, header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

    # STEP = 500
    # SUB_STEP = 100
    # for i in range(SUB_STEP, epochs, SUB_STEP):
    #     q = epochs // STEP
    #     folder = 'W-F-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str((q + 1) * 500) + '-eta=' + str(eta)
    #     sub_folder = 'W-F-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(i) + '-eta=' + str(eta)
    #     weight_and_biases_path = folder + '/' + sub_folder

    accuracy = np.zeros(epochs // 100)
    for i in range(500, epochs + 500, 500):
        parent_folder = 'F-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta) + '-epoch=' + str(i)
        for j in range(i - 400, i + 100, 100):
            if j > epochs:
                break
            child_folder = 'epoch=' + str(j)
            weight_and_biases_path = parent_folder + '/' + child_folder

            w = pd.read_csv('forward/training/' + weight_and_biases_path + '/W.csv', header=None)
            W = w.to_numpy()

            b = pd.read_csv('forward/training/' + weight_and_biases_path + '/B.csv', header=None)
            B = b.to_numpy()

            # Bias learnable
            # Expand matrix W with a column B
            WB = np.insert(W, W.shape[1], np.transpose(B), axis=1)

            # Expand matrix X with a column of 1s
            XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

            percentage, error_label, images, error_output_nn = forward_accuracy(YV, WB, XV_hat, validation_threshold)

            accuracy[(j // 100) - 1] = percentage

            # print('Validation: Done')

            # display_validation(percentage, error_label, images, error_output_nn)

    # plot
    x = np.arange(100, epochs + 100, 100)
    y = accuracy
    plt.plot(x, y, color='cyan')

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    plt.title('Threshold: ' + str(int(validation_threshold*100)) + '%')
    annotation_string = (r'$\eta$ = ' + str(eta) + '\n'
                         + '#Patterns = ' + str(pattern) + '\n'
                         + 'Learning mode = ' + learning_mode + '\n')

    plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

    parent_figure_folder = ('validation/figure-forward/' + str(validation_dataset_name) + '/threshold='
                            + str(validation_threshold) + '/')

    parent_figure_folder = os.path.join(os.getcwd(), parent_figure_folder)

    child_figure_folder = (parent_figure_folder + '/F-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta)
                          + '-epoch=' + str(epochs))

    if not os.path.exists(parent_figure_folder):
        # Create the folder if it doesn't exist
        os.makedirs(parent_figure_folder)
    plt.savefig(child_figure_folder + '.png')

    plt.show()


def backprop_validation(validation_dataset_path, learning_mode, pattern, epochs, eta, validation_threshold):
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

            W = []
            B = []

            for i in range(0, 3):
                b = pd.read_csv('backpropagation/training/' + weight_and_biases_path + '/' + 'B' + str(i) + '.csv',
                                header=None)
                b = b[0].to_numpy()
                B.append(b.reshape(-1, 1))

                w = pd.read_csv('backpropagation/training/' + weight_and_biases_path + '/' + 'W' + str(i) + '.csv',
                                header=None)
                W.append(w.to_numpy())

            percentage, error_label, images, error_output_nn = backprop_accuracy(YV, W, XV, B, validation_threshold)

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
