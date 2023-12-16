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
l = 60000

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

# TODO refactor with a method that pass a matrix nx2 that specify the layout of the NN

# W is a matrix
W = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, INPUT_DIMENSION))

# B Vector
B = np.full((OUTPUT_DIMENSION, 1), -10.)

print('---------- Training ----------')

X = input_normalization_Matrix(X)
epochs = 100
#learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

YB, XB = learning_method(Y, X, learning_mode, 100)
for i in range(0, len(XB)):
    batch_iteration = i
    W = gradient_descent_algorithm(YB[i], W, XB[i], B, ETA, epochs, batch_iteration)

#W = gradient_descent_algorithm(Y, W, X, B, ETA, epochs)
weight = pd.DataFrame(W)

# W-1L-batch-epochs
file_name = 'W-1L-F-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs)

weight.to_csv('weight-csv/' + file_name + '.csv', encoding='utf-8', header=False, index=False)
