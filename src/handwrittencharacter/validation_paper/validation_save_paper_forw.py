import os

import numpy as np
import pandas as pd

from src.handwrittencharacter.lib.forward.accuracy import accuracy as forward_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix


def forward(PATH_MAIN_FILE, validation_dataset_name, weight_and_biases_path, epochs, STEP, validation_threshold):
    print('Validation: Start')

    folders = weight_and_biases_path.split('/')

    parameters = folders[0].split('-')

    pattern = parameters[2].split('=')[1]
    eta = parameters[3].split('=')[1]
    learning_mode = parameters[1]

    parent_folder = 'F-' + str(learning_mode) + '-l=' + str(pattern) + '-eta=' + str(eta)

    if learning_mode != 'batch' and learning_mode != 'online':
        learning_mode = parameters[1].split('=')[0]
        batch_dimension = int(parameters[1].split('=')[1])

        parent_folder = ('F-' + str(learning_mode) + '=' + str(batch_dimension) + '-l=' + str(pattern) + '-eta=' +
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
        w = pd.read_csv(PATH_MAIN_FILE + '/forward/training/' + weight_and_biases_path + '/W.csv', header=None)
        W = w.to_numpy()

        # Biases
        b = pd.read_csv(PATH_MAIN_FILE + '/forward/training/' + weight_and_biases_path + '/B.csv', header=None)
        B = b.to_numpy()

        # Expand matrix W with a column B
        WB = np.insert(W, W.shape[1], np.transpose(B), axis=1)

        # Expand matrix X with a column of 1s
        XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

        percentage, error_label, images, error_output_nn = forward_accuracy(YV, WB, XV_hat, validation_threshold)

        accuracy[(i // STEP) - 1] = percentage

        progressing = i / epochs * 100
        print('Validation graph progress: ' + str(int(progressing)) + ' %')

    x = np.arange(STEP, epochs + STEP, STEP)
    y = accuracy

    return x, y


# Save CSV for Excel export and report

PATH_MAIN_FILE = os.path.dirname(__file__)

epochs = 2000  # epochs
STEP = 100  # for validation graph

# validation_dataset_name = 'mnist_test'

validation_dataset_name = 'salt_pepper/mnist_test-sp-s-1-p-0.6'

validation_threshold_array = [0.0, 0.5, 0.9]

nn_array = [
    'F-batch-l=60000-eta=0.01',
    'F-batch-l=60000-eta=0.001',
    'F-batch-l=60000-eta=0.0001',
    'F-batch-l=60000-eta=0.00001',

    'F-mini=128-l=60000-eta=0.1',
    'F-mini=128-l=60000-eta=0.01',
    'F-mini=128-l=60000-eta=0.001',
    'F-mini=128-l=60000-eta=0.0001',

    'F-mini=256-l=60000-eta=0.1',
    'F-mini=256-l=60000-eta=0.01',
    'F-mini=256-l=60000-eta=0.001',
    'F-mini=256-l=60000-eta=0.0001',

    'F-mini=512-l=60000-eta=0.1',
    'F-mini=512-l=60000-eta=0.01',
    'F-mini=512-l=60000-eta=0.001',
    'F-mini=512-l=60000-eta=0.0001',

    'F-mini=1024-l=60000-eta=0.1',
    'F-mini=1024-l=60000-eta=0.01',
    'F-mini=1024-l=60000-eta=0.001',
    'F-mini=1024-l=60000-eta=0.0001',

    'F-online-l=60000-eta=0.1',
    'F-online-l=60000-eta=0.01',
    'F-online-l=60000-eta=0.001',
    'F-online-l=60000-eta=0.0001']

import csv

for nn in nn_array:
    for threshold in validation_threshold_array:
        validation_threshold = threshold
        weight_and_biases_path = nn

        epochs_array, accuracy_array = forward(PATH_MAIN_FILE + '/../', validation_dataset_name, weight_and_biases_path,
                                               epochs,
                                               STEP, validation_threshold)

        print(epochs_array)
        print(accuracy_array)

        # Data to append
        row1 = [weight_and_biases_path, 'validation_threshold=' + str(validation_threshold)]
        row2 = epochs_array
        row3 = accuracy_array

        # Specify the file name
        filename = "F-data-" + validation_dataset_name + ".csv"

        # Open the file in 'a' mode to append
        with open(PATH_MAIN_FILE + '/' + filename, 'a', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Append the rows
            csvwriter.writerow(row1)
            csvwriter.writerow(row2)
            csvwriter.writerow(row3)

        print("Data appended to CSV file successfully!")
