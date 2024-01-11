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

pixelHeight = 28
pixelWidth = 28

# image = X[0].reshape(28, 28)
# plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.show()

percentage = 0.2

# pi*r^2 = 10% * 28*28
radiusBig = int(np.sqrt(28*28*percentage/math.pi))

for l in range(0, len(X)):
    image = X[l].reshape(28, 28)
    blobNumber = np.random.randint(2, 6)
    remainRadius = radiusBig
    for b in range(0, blobNumber):
        if b == blobNumber-1:
            radius = remainRadius
        else:
            if remainRadius*0.6 < 3:
                radius = np.random.randint(2, 3)
            else:
                radius = np.random.randint(2, remainRadius * 0.6)
            remainRadius = remainRadius - radius

        center = np.random.randint(28, size=2)
        # blob creation
        for i in range(-radius, radius+1):
            if 0 <= center[0] + i < 28:
                for j in range(-radius, radius+1):
                    if 0 <= center[1] + j < 28:
                        if np.sqrt(i**2 + j**2) <= radius:
                            image[center[0] + i][center[1] + j] = 255

    X[l] = image.reshape(1, -1)

# image = X[0].reshape(28, 28)
# plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.show()

bl = pd.DataFrame(np.insert(X, 0, Y, axis=1))

new_file_name = file_name + '-' + 'bl-p-' + str(percentage)

folder_path = '../../../dataset/blob/'

if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

bl.to_csv(folder_path + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
