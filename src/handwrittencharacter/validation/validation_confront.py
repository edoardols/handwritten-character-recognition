# region IMPORTS
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # Usa il backend TkAgg
from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.backprop.accuracy import accuracy as backpropagation_accuracy
from src.handwrittencharacter.lib.forward.accuracy import accuracy as forward_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix
# endregion


def validation_confront_graph(PATH_MAIN_FILE, validation_dataset_name, nn_array, epochs, STEP, validation_threshold):

    coordinates = []
    str_legend = []
    for i in range(0, len(nn_array)):
        weight_and_biases_path = nn_array[i]

        if 'F' in weight_and_biases_path:
            x, y = forward(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, 100,
                                       validation_threshold)
            matrix = np.concatenate((x, y), axis=0)
            coordinates.append(matrix)
        elif 'B' in weight_and_biases_path:
            x, y = backprop(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, 50,
                            validation_threshold)
            matrix = np.concatenate((x, y), axis=0)
            coordinates.append(matrix)
        else:
            print('Something went wrong...')
            exit()

        plt.plot(x, y)
        str_legend.append(nn_array[i])

    plt.xlabel('Epochs')
    plt.ylabel('Empirical risk')
    plt.title('Threshold: ' + str(int(validation_threshold * 100)) + '%')
    plt.legend(str_legend)

    plt.show()

# region FUNCTIONS


def forward(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, STEP, validation_threshold):

    print('Validation with ' + validation_dataset_name + ': Start')

    folders = weight_and_biases_path.split('/')

    parameters = folders[0].split('-')

    pattern = parameters[2].split('=')[1]
    eta = parameters[3].split('=')[1]
    learning_mode = parameters[1]

    validation_dataset_path = PATH_MAIN_FILE + '/../../dataset/' + validation_dataset_name + '.csv'
    validation_dataset = pd.read_csv(validation_dataset_path, header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

    accuracy = np.zeros(epochs // STEP)
    parent_folder = 'F-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta)
    for i in range(STEP, epochs + STEP, STEP):
        child_folder = 'epoch=' + str(i)
        weight_and_biases_path = parent_folder + '/' + child_folder

        w = pd.read_csv(PATH_MAIN_FILE + '/forward/training/' + weight_and_biases_path + '/W.csv', header=None)
        W = w.to_numpy()

        b = pd.read_csv(PATH_MAIN_FILE + '/forward/training/' + weight_and_biases_path + '/B.csv', header=None)
        B = b.to_numpy()

        # Bias learnable
        # Expand matrix W with a column B
        WB = np.insert(W, W.shape[1], np.transpose(B), axis=1)

        # Expand matrix X with a column of 1s
        XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

        percentage, error_label, images, error_output_nn = forward_accuracy(YV, WB, XV_hat, validation_threshold)

        accuracy[(i // STEP) - 1] = percentage

        # print('Validation: Done')

        # display_validation(percentage, error_label, images, error_output_nn)

    # plot
    x = np.arange(STEP, epochs + STEP, STEP)
    y = accuracy

    return x, y


def backprop(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, STEP, validation_threshold):
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

        progressing = i/epochs * 100
        print('Validation graph progress: ' + str(int(progressing)) + ' %')
    # endregion

    # region PLOT
    x = np.arange(STEP, epochs + STEP, STEP)
    y = accuracy

    return x, y
# endregion
