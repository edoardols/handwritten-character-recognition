import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.forward.accuracy import accuracy as forward_accuracy
from src.handwrittencharacter.lib.backprop.accuracy import accuracy as backpropagation_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix
from src.handwrittencharacter.validation.display_validation import display_validation

global current_index
current_index = 0


def backpropagation_validation_single(PATH_MAIN_FILE, validation_dataset_path, weight_and_biases_path, validation_threshold):
    print('Validation: Start')
    validation_dataset = pd.read_csv(PATH_MAIN_FILE + '/../../dataset/' + validation_dataset_path + '.csv', header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

    path = PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/'

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

    # Expand matrix X with a column of 1s
    XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

    percentage, error_label, images, error_output_nn = backpropagation_accuracy(YV, WB0, WB1, WB2, XV_hat, validation_threshold)

    print('Validation: Done')

    display_validation(percentage, error_label, images, error_output_nn)


def backpropagation_validation_graph(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, STEP,
                                     validation_threshold):
    print('Validation: Start')

    folders = weight_and_biases_path.split('/')

    parameters = folders[0].split('-')

    pattern = parameters[2].split('=')[1]
    eta = parameters[3].split('=')[1]
    learning_mode = parameters[1]

    parent_folder = 'B-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta)

    if learning_mode != 'batch' and learning_mode != 'online':
        learning_mode = parameters[1].split('=')[0]
        batch_dimension = int(parameters[1].split('=')[1])

        parent_folder = ('B-' + str(learning_mode) + '=' + str(batch_dimension) + '-l=' + str(pattern) + '-eta=' +
                         str(eta))

    validation_dataset_path = PATH_MAIN_FILE + '/../../dataset/' + validation_dataset_name + '.csv'
    validation_dataset = pd.read_csv(validation_dataset_path, header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

    accuracy = np.zeros(epochs // STEP)

    # region VALIDATION
    for i in range(STEP, epochs + STEP, STEP):
        child_folder = 'epoch=' + str(i)
        weight_and_biases_path = parent_folder + '/' + child_folder

        # Weights
        w = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/W0.csv', header=None)
        W0 = w.to_numpy()
        w = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/W1.csv', header=None)
        W1 = w.to_numpy()
        w = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/W2.csv', header=None)
        W2 = w.to_numpy()

        # Biases
        b = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/B0.csv', header=None)
        B0 = b.to_numpy()
        b = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/B1.csv', header=None)
        B1 = b.to_numpy()
        b = pd.read_csv(PATH_MAIN_FILE + '/backpropagation/training/' + weight_and_biases_path + '/B2.csv', header=None)
        B2 = b.to_numpy()

        # Expand matrix W with a column B
        WB0 = np.insert(W0, W0.shape[1], np.transpose(B0), axis=1)
        WB1 = np.insert(W1, W1.shape[1], np.transpose(B1), axis=1)
        WB2 = np.insert(W2, W2.shape[1], np.transpose(B2), axis=1)

        # Expand matrix X with a column of 1s
        XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

        percentage, error_label, images, error_output_nn = backpropagation_accuracy(YV, WB0, WB1, WB2, XV_hat,
                                                                                    validation_threshold)

        accuracy[(i // STEP) - 1] = percentage

        progressing = i/(epochs) * 100
        print('Validation graph progress: ' + str(int(progressing)) + ' %')
    # endregion

    # region PLOT
    x = np.arange(STEP, epochs + STEP, STEP)
    y = accuracy
    plt.plot(x, y, color='cyan')

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    plt.title('Threshold: ' + str(int(validation_threshold * 100)) + '%')
    annotation_string = (r'$\eta$ = ' + str(eta) + '\n'
                         + '#Patterns = ' + str(pattern) + '\n'
                         + 'Learning mode = ' + learning_mode + '\n')

    plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')
    # endregion

    child_figure_folder = PATH_MAIN_FILE + '/backpropagation/training/' + parent_folder + '/validation'

    plt.savefig(child_figure_folder + '.png')

    plt.show()