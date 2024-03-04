import pandas as pd
import numpy as np

import os

from matplotlib import pyplot as plt

file_name = 'mnist_test'
dataset = pd.read_csv('../../../dataset/' + file_name + '.csv', header=None)

# Number of examples

X_D = dataset.iloc[:, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:, :1]
Y = Y_D[0].to_numpy()

image = X[0].reshape(28, 28)

pixelHeight = 28
pixelWidth = 28

image = X[3].reshape(28, 28)

plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.show()

percentage = 0.3

for l in range(0, len(X)):
    # if l == 1:
    #     break
    X[l] = X[l]*percentage

image = X[3].reshape(28, 28)
plt.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.show()

br = pd.DataFrame(np.insert(X, 0, Y, axis=1))

new_file_name = file_name + '-' + 'br-p=' + str(percentage)

folder_path = '../../../dataset/brightness/'

if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

br.to_csv(folder_path + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
