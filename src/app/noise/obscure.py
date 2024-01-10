import math

import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

file_name = 'mnist_test'
dataset = pd.read_csv('../../../data/' + file_name + '.csv', header=None)

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

np.random.randint(2, 6)

# image = X[0].reshape(28, 28)
# plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
# plt.show()

ob = pd.DataFrame(np.insert(X, 0, Y, axis=1))

new_file_name = file_name + '-' + 'th-step=' + str(step)

folder_path = '../../../data/thickness/'

if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

ob.to_csv(folder_path + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
