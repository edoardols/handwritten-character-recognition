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

image = X[0].reshape(28, 28)

pixelHeight = 28
pixelWidth = 28

#plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
#plt.show()

# -255 to 255

#A = np.random.uniform(low=-255, high=255, size=(pixelHeight, pixelWidth))


percentage = 0.6
pixelPercentage = int(percentage*(pixelHeight*pixelWidth))

severity = 1


def bound(pixel):
    if pixel > 255:
        return 255
    if pixel < 0:
        return 0
    return pixel


for l in range(0, len(X)):
    image = X[l].reshape(28, 28)
    for p in range(0, pixelPercentage):
        i = np.random.randint(0, pixelHeight)
        j = np.random.randint(0, pixelWidth)
        pixel = np.random.uniform(low=int(-255*severity), high=int(255*severity))
        image[i][j] = bound(image[i][j] + pixel)
    X[l] = image.reshape(1, -1)

plt.imshow(255 - image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.show()

# sp = pd.DataFrame(np.insert(X, 0, Y, axis=1))

# W-1L-batch-epochs
# new_file_name = file_name + '-' + 'sp-s-' + str(severity) + '-p-' + str(percentage)

# sp.to_csv('../../../dataset/salt_pepper/' + new_file_name + '.csv', encoding='utf-8', header=False, index=False)
