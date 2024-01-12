import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.lib.forward.accuracy import accuracy as forward_accuracy
from src.lib.backprop.accuracy import accuracy as backprop_accuracy
from src.lib.mapping import input_normalization_Matrix

global current_index
current_index = 0

def forward_validation(validation_dataset_path, learning_mode, pattern, epochs, eta, validation_threshold):
    print('Validation: Start')

    validation_dataset = pd.read_csv('../../dataset/' + validation_dataset_path + '.csv', header=None)

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
        parent_folder = 'W-F-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(i) + '-eta=' + str(eta)
        for j in range(i - 400, i + 100, 100):
            if j > epochs:
                break
            child_folder = 'W-F-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(j) + '-eta=' + str(eta)
            weight_and_biases_path = parent_folder + '/' + child_folder

            w = pd.read_csv('forward/weight-csv/' + weight_and_biases_path + '/W.csv', header=None)
            W = w.to_numpy()

            b = pd.read_csv('forward/weight-csv/' + weight_and_biases_path + '/B.csv', header=None)
            B = b.to_numpy()

            percentage, error_label, images, error_output_nn = forward_accuracy(YV, W, XV, B, validation_threshold)

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

    path_figure_folder = 'validation/figure-forward/' + validation_dataset + '/threshold=' + str(validation_threshold) + '/W-F-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(epochs) + '-eta=' + str(eta)

    if not os.path.exists('validation/figure-forward/threshold=' + str(validation_threshold)):
        # Create the folder if it doesn't exist
        os.makedirs('validation/figure-forward/threshold=' + str(validation_threshold))
    plt.savefig(path_figure_folder + '.png')


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
        parent_folder = 'W-B-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(i) + '-eta=' + str(eta)
        for j in range(i - 400, i + 100, 100):
            if j > epochs:
                break
            child_folder = 'W-B-' + str(learning_mode) + '-l=' + str(pattern) + '-epoch=' + str(j) + '-eta=' + str(eta)
            weight_and_biases_path = parent_folder + '/' + child_folder

            W = []
            B = []

            for i in range(0, 3):
                b = pd.read_csv('backpropagation/weight-csv/' + weight_and_biases_path + '/' + 'B' + str(i) + '.csv',
                                header=None)
                b = b[0].to_numpy()
                B.append(b.reshape(-1, 1))

                w = pd.read_csv('backpropagation/weight-csv/' + weight_and_biases_path + '/' + 'W' + str(i) + '.csv',
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
