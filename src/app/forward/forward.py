import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from src.lib.mapping import input_normalization_Matrix
from src.lib.forward.gradient import gradient_descent_algorithm

print('Loading dataset: Start')

dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# Number of examples
l = 505
dataset = dataset.iloc[:l, :]

# X_D = dataset.iloc[:l, 1:]
# X = X_D.to_numpy()
#
# X = input_normalization_Matrix(X)
#
# Y_D = dataset.iloc[:l, :1]
# Y = Y_D[0].to_numpy()

print('Loading dataset: Done')

print('Neural Network: Start')

# configuration file
# OLNN have structure [num:output]
OLNN = pd.read_csv('OLNN.csv')

# dim input
# INPUT_DIMENSION = len(X[0])
INPUT_DIMENSION = 28*28

# learning rate
ETA = 0.01

# output dim
OUTPUT_DIMENSION = OLNN.iloc[0, 1]

# Layout Neural Network
np.random.seed(42)

# W is a matrix
W = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, INPUT_DIMENSION))

# B Vector
B = np.full((OUTPUT_DIMENSION, 1), -10.)

print('Neural Network: Done')

print('Training: Start')

epochs = 100
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

E = np.zeros(epochs, dtype=float)

for e in range(0, epochs):
    # W, E_epoch = gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode)
    W, E_epoch = gradient_descent_algorithm(dataset, W, B, ETA, e, learning_mode)
    E[e] = E_epoch

print('Training: Done')

print('Saving: Start')

# W = gradient_descent_algorithm(Y, W, X, B, ETA, epochs)
weight = pd.DataFrame(W)

# W-1L-batch-epochs
file_name = 'W-1L-F-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs)

weight.to_csv('weight-csv/' + file_name + '.csv', encoding='utf-8', header=False, index=False)

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

plt.savefig('weight-csv/' + file_name)

# plt.show()

print('Saving: Done')
