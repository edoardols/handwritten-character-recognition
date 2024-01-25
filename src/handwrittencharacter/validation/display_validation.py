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