import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.handwrittencharacter.lib.forward.accuracy import accuracy as forward_accuracy
from src.handwrittencharacter.lib.backprop.accuracy import accuracy as backpropagation_accuracy
from src.handwrittencharacter.lib.mapping import input_normalization_Matrix

global current_index
current_index = 0


def display_validation(percentage, error_label, images, error_output_nn):

    fig, ax = plt.subplots()

    # Display the first image
    img_plot = ax.imshow(1 - images[0].reshape(28, 28), cmap='gray')

    annotation_string = ('Label = ' + str(error_label[0]) + '\n'
                         + 'Output = ' + str(error_output_nn[0]) + '\n'
                         + 'Error: ' + str(1) + ' of ' + str(len(images)))

    annotation = ax.annotate(annotation_string, xy=(0.79, 0.13), xycoords='figure fraction',
                             horizontalalignment='right')

    def on_arrow_key(event, ):
        global current_index
        if event.key == 'right':
            current_index = (current_index + 1) % len(images)
        elif event.key == 'left':
            current_index = (current_index - 1) % len(images)

        img_plot.set_data(1 - images[current_index].reshape(28, 28))

        annotation.set_text('Label = ' + str(error_label[current_index]) + '\n'
                            + 'Output = ' + str(error_output_nn[current_index]) + '\n'
                            + 'Error: ' + str(current_index + 1) + ' of ' + str(len(images)))
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_arrow_key)
    plt.title('Accuracy: ' + str(percentage) + '%')
    plt.show()

    data = error_label

    # Create a histogram
    plt.hist(data, bins=np.arange(1, 11), align='left', rwidth=0.8, alpha=0.7)

    plt.xlabel('Numbers')
    plt.ylabel('Frequency')
    plt.title('Error distribution')
    plt.xticks(np.arange(1, 10))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def forward_validation_single(validation_dataset_path, weight_and_biases_path, validation_threshold):
    print('Validation: Start')
    validation_dataset = pd.read_csv('../../dataset/' + validation_dataset_path + '.csv', header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

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

    print('Validation: Done')

    display_validation(percentage, error_label, images, error_output_nn)


def backpropagation_validation_single(validation_dataset_path, weight_and_biases_path, validation_threshold):
    print('Validation: Start')
    validation_dataset = pd.read_csv('../../dataset/' + validation_dataset_path + '.csv', header=None)

    XV_D = validation_dataset.iloc[:, 1:]
    XV = XV_D.to_numpy()

    YV_D = validation_dataset.iloc[:, :1]
    YV = YV_D[0].to_numpy()

    XV = input_normalization_Matrix(XV)

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

    # Expand matrix X with a column of 1s
    XV_hat = np.insert(XV, XV.shape[1], np.transpose(np.ones((XV.shape[0], 1), dtype=float)), axis=1)

    percentage, error_label, images, error_output_nn = backpropagation_accuracy(YV, WB0, WB1, WB2, XV_hat, validation_threshold)

    print('Validation: Done')

    display_validation(percentage, error_label, images, error_output_nn)

