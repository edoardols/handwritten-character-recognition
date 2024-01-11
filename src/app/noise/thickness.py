import math
import copy

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

pixelHeight = 28
pixelWidth = 28

# image = X[12].reshape(28, 28)
# plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.show()

step = -3

for l in range(0, len(X)):
    image = X[l].reshape(28, 28)
    newImage = copy.deepcopy(image)
    if step > 0:
        for s in range(0, step):
            for i in range(0 + 1, 28 - 1):
                for j in range(0 + 1, 28 - 1):
                    # j-1 left
                    if image[i][j - 1] < image[i][j]:
                        newImage[i][j - 1] = image[i][j]
                    # i-1 up
                    if image[i - 1][j] < image[i][j]:
                        newImage[i - 1][j] = image[i][j]
                    # j+1 right
                    if image[i][j + 1] < image[i][j]:
                        newImage[i][j + 1] = image[i][j]
                    # i+1 down
                    if image[i + 1][j] < image[i][j]:
                        newImage[i + 1][j] = image[i][j]
            image = copy.deepcopy(newImage)

    elif step < 0:
        for s in range(0, -step):
            for i in range(0 + 1, 28 - 1):
                for j in range(0 + 1, 28 - 1):
                    # j-1 left
                    if image[i][j - 1] < image[i][j]:
                        newImage[i][j] = image[i][j - 1]
                    # i-1 up
                    if image[i - 1][j] < image[i][j]:
                        newImage[i][j] = image[i - 1][j]
                    # j+1 right
                    if image[i][j + 1] < image[i][j]:
                        newImage[i][j] = image[i][j + 1]
                    # i+1 down
                    if image[i + 1][j] < image[i][j]:
                        newImage[i][j] = image[i + 1][j]

            image = copy.deepcopy(newImage)

    X[l] = newImage.reshape(1, -1)

# image = X[12].reshape(28, 28)
# plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.show()

if step != 0:
    th = pd.DataFrame(np.insert(X, 0, Y, axis=1))

    new_file_name = file_name + '-' + 'th-step=' + str(step)

    folder_path = '../../../dataset/thickness/'

    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)

    th.to_csv(folder_path + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
