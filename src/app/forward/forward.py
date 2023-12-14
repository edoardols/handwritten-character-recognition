import pandas as pd
import numpy as np

from src.lib.learning_method import learning_method
from src.lib.mapping import input_normalization_Matrix
from src.lib.forward.gradient import gradient_descent_algorithm

# configuration file
# OLNN have structure [num:output]
OLNN = pd.read_csv('OLNN.csv')

dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# Number of examples
l = 6000

X_D = dataset.iloc[:l, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:l, :1]
Y = Y_D[0].to_numpy()

# dim input
INPUT_DIMENSION = len(X[0])

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

print('---------- Training ----------')

X = input_normalization_Matrix(X)
epochs = 1000
learning_mode = 'batch'
#learning_mode = 'mini'
# learning_mode = 'online'

# TODO change
#XB = learning_method(X, learning_mode)

#for i in range(0, XB.shape(1)):
#W = gradient_descent_algorithm(Y, W, XB[i], B, ETA, epochs)

W = gradient_descent_algorithm(Y, W, X, B, ETA, epochs)
weight = pd.DataFrame(W)

# W-1L-batch-epochs
file_name = 'W-1L-F-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs)

weight.to_csv('weight-csv/' + file_name + '.csv', encoding='utf-8', header=False, index=False)
