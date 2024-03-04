import os

import numpy as np
import pandas as pd

from src.handwrittencharacter.lib.backprop.accuracy import accuracy as backpropagation_accuracy
from src.handwrittencharacter.lib.forward.accuracy import accuracy as forward_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix


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

        progressing = i / epochs * 100
        print('Validation graph progress: ' + str(int(progressing)) + ' %')

    x = np.arange(STEP, epochs + STEP, STEP)
    y = accuracy

    return x, y


# Save CSV for Excel export and report

PATH_MAIN_FILE = os.path.dirname(__file__)

epochs = 500  # epochs
STEP = 10  # for validation graph

validation_dataset_name = 'mnist_test'

validation_threshold_array = [0.0, 0.5, 0.9]

nn_array = [
    #'B-batch-l=60000-eta=0.01',
    #'B-batch-l=60000-eta=0.001',
    #'B-batch-l=60000-eta=0.0001',

    'B-mini=128-l=60000-eta=0.1',
    #'B-mini=128-l=60000-eta=0.001',
    'B-mini=128-l=60000-eta=0.0001',

    #'B-mini=256-l=60000-eta=0.001',

    #'B-mini=512-l=60000-eta=0.01',
    #'B-mini=512-l=60000-eta=0.001',
    #'B-mini=512-l=60000-eta=0.0001',

    'B-mini=1024-l=60000-eta=0.01',
    #'B-mini=1024-l=60000-eta=0.001',
    'B-mini=1024-l=60000-eta=0.0001']

#     'B-mini=512-l=60000-eta=0.1'
#     'B-mini=256-l=60000-eta=0.0001',
#     'B-mini=256-l=60000-eta=0.1',
#     'B-mini=256-l=60000-eta=0.01',
# 'F-mini=128-l=60000-eta=0.01',

import csv

for nn in nn_array:
    for threshold in validation_threshold_array:

        validation_threshold = threshold
        weight_and_biases_path = nn

        epochs_array, accuracy_array = backprop(PATH_MAIN_FILE + '/../', validation_dataset_name, weight_and_biases_path, epochs, STEP, validation_threshold)

        print(epochs_array)
        print(accuracy_array)

        # Data to append
        row1 = [weight_and_biases_path, 'validation_threshold=' + str(validation_threshold)]
        row2 = epochs_array
        row3 = accuracy_array

        # Specify the file name
        filename = "B-data-" + validation_dataset_name + ".csv"

        # Open the file in 'a' mode to append
        with open(PATH_MAIN_FILE + '/' + filename, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Append the rows
            csvwriter.writerow(row1)
            csvwriter.writerow(row2)
            csvwriter.writerow(row3)

        print("Data appended to CSV file successfully!")