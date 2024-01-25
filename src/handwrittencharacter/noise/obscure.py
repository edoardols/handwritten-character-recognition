import math

import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

file_name = 'mnist_test'
dataset = pd.read_csv('../../../dataset/' + file_name + '.csv', header=None)

# Number of examples
X_D = dataset.iloc[:, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:, :1]
Y = Y_D[0].to_numpy()

X_CENTER = 13
Y_CENTER = 13

X_changed = X * 0
for l in range(0, len(X)):

    images = X[l].reshape(28, 28)
    obscured_part = X_changed[l].reshape(28, 28)

    # # Find indices where the value is greater than 0
    # pattern_pixels = np.where(image > 0)
    #
    # # Randomly select an index from the list of non-zero indices
    # random_index = np.random.choice(len(pattern_pixels[0]))
    #
    # x_pixel = pattern_pixels[0][random_index]
    # y_pixel = pattern_pixels[1][random_index]

    list_pattern_pixel = list(zip(*np.where(images > 0)))

    index = np.random.randint(0, len(list_pattern_pixel))

    chosen_pixel = list_pattern_pixel[index]

    x_pixel_pattern = chosen_pixel[0]
    y_pixel_pattern = chosen_pixel[1]

    # Get point on the edge
    edge = np.random.randint(0, 4)

    if edge == 0:  # upper edge
        x_edge_point = np.random.randint(0, 28)
        y_edge_point = 0
    elif edge == 1:  # right edge
        x_edge_point = 27
        y_edge_point = np.random.randint(0, 28)
    elif edge == 2:  # bottom edge
        x_edge_point = np.random.randint(0, 28)
        y_edge_point = 27
    else:  # edge == 3  # left edge
        x_edge_point = np.random.randint(0, 28)
        y_edge_point = 0

    def line(x):
        # (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)

        # pixel_pattern as (x1, y1)
        x1 = x_pixel_pattern
        y1 = y_pixel_pattern

        # edge_point as (x2, y2)
        x2 = x_edge_point
        y2 = y_edge_point

        # line is parallel
        if x2 - x1 == 0:
            return 0

        y = ((x - x1) / (x2 - x1)) * (y2 - y1) + y1
        return y


    for i in range(0, 28):
        for j in range(0, 28):
            # The line divide the pattern in two part
            # We get the smaller part (the center is not comprised)
            # If the pixel is in the smaller part we change the color to 0

            # Check if the part is the smaller

            if Y_CENTER <= line(X_CENTER) and j >= line(i):
                obscured_part[i][j] = images[i][j]
                images[i][j] = 0

            if Y_CENTER >= line(X_CENTER) and j <= line(i):
                obscured_part[i][j] = images[i][j]
                images[i][j] = 0

    # X[l] = images
    X[l] = images.reshape(1, -1)
    X_changed[l] = obscured_part.reshape(1, -1)

fig, ax = plt.subplots()

# Display the first image

img_plot_gray = ax.imshow(255 - X[0].reshape(28, 28), cmap='gray')

# Create a mask for pixels greater than a threshold
threshold = 1
mask = X_changed[0].reshape(28, 28) > threshold

img_plot_blue = ax.imshow(np.ma.array(X_changed[0].reshape(28, 28), mask=~mask), cmap='Blues')

# annotation_string = ('Label = ' + str(error_label[0]) + '\n'
#                      + 'Output = ' + str(error_output_nn[0]) + '\n'
#                      + 'Error: ' + str(1) + ' of ' + str(len(images)))
#
# annotation = ax.annotate(annotation_string, xy=(0.79, 0.13), xycoords='figure fraction',
#                          horizontalalignment='right')

current_index = 0


def on_arrow_key(event, ):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(X)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(X)

    mask = X_changed[current_index].reshape(28, 28) > threshold
    img_plot_blue.set_data(np.ma.array(X_changed[current_index].reshape(28, 28), mask=~mask))
    img_plot_gray.set_data(255 - X[current_index].reshape(28, 28))

    # annotation.set_text('Label = ' + str(error_label[current_index]) + '\n'
    #                     + 'Output = ' + str(error_output_nn[current_index]) + '\n'
    #                     + 'Error: ' + str(current_index + 1) + ' of ' + str(len(images)))
    fig.canvas.draw()


fig.canvas.mpl_connect('key_press_event', on_arrow_key)
# plt.title('Accuracy: ' + str(percentage) + '%')
plt.show()

ob = pd.DataFrame(np.insert(X, 0, Y, axis=1))

new_file_name = file_name + '-' + 'ob'

folder_path = '../../../dataset/obscure/'

if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

ob.to_csv(folder_path + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
