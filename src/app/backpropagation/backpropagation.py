import pandas as pd
import numpy as np

import os

from matplotlib import pyplot as plt

from src.lib.backprop.gradient import gradient_descent_algorithm
from src.lib.mapping import input_normalization_Matrix

print('Loading dataset: Start')

dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# Number of examples
l = 600

X_D = dataset.iloc[:l, 1:]
X = X_D.to_numpy()

X = input_normalization_Matrix(X)

Y_D = dataset.iloc[:l, :1]
Y = Y_D[0].to_numpy()

print('Loading dataset: Done')

print('Neural Network: Start')

# configuration file
# HLNN have structure [num_layer][num_neuron]
HLNN = pd.read_csv('../forward/HLNN.csv')
# OLNN have structure [num:output]
OLNN = pd.read_csv('../forward/OLNN.csv')

# dim input
INPUT_DIMENSION = len(X[0])

# learning rate
ETA = 0.01

# output dim
OUTPUT_DIMENSION = OLNN.iloc[0, 1]

# Layout Neural Network
np.random.seed(42)

# Layout Neural Network
# W is a list of matrix
W = []

# TODO refactor with a method that pass a matrix nx2 that specify the layout of the NN
for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[0, 1], INPUT_DIMENSION))
    else:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[i, 1], HLNN.iloc[i - 1, 1]))
    W.append(w)

w = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, HLNN.iloc[HLNN.shape[0]-1, 1]))
W.append(w)

# B is a list of vector
B = []

for i in range(HLNN.shape[0]):
    b = np.full((HLNN.iloc[i, 1], 1), 0.)
    B.append(b)

b = np.full((OUTPUT_DIMENSION, 1), 0.)
B.append(b)

print('Neural Network: Done')

print('Training: Start')

epochs = 100
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

E = np.zeros(epochs, dtype=float)

for e in range(0, epochs):
    W, E_epoch = gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode)
    E[e] = E_epoch

print('Training: Done')

print('Saving: Start')

folder_name = 'W-BP-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs) + '/'

folder_path = os.path.join(os.getcwd(), 'weight-csv/' + folder_name)
# Check if the folder already exists
if not os.path.exists(folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path)

for i in range(0, len(W)):
    # 0 is the input layer
    weight = pd.DataFrame(W[i])
    weight.to_csv(folder_path + 'W' + str(i) + '.csv', encoding='utf-8', header=False, index=False)

for i in range(0, len(B)):
    # 0 is the input layer
    bias = pd.DataFrame(B[i])
    bias.to_csv(folder_path + 'B' + str(i) + '.csv', encoding='utf-8', header=False, index=False)

# plot
x = np.arange(0, epochs, 1)
y = E
plt.plot(x, y)

plt.xlabel('Epochs')
plt.ylabel('Empirical risk')
annotation_string = (r'$\eta$ = ' + str(ETA) + '\n'
                     + '#Patterns = ' + str(l) + '\n'
                     + 'Learning mode = ' + learning_mode + '\n')

plt.annotate(annotation_string, xy=(0.88, 0.72), xycoords='figure fraction', horizontalalignment='right')

plt.savefig('weight-csv/' + folder_name + 'E')

print('Saving: Done')
