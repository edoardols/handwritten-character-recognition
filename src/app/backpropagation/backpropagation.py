import pandas as pd
import numpy as np

from src.lib.backprop.accuracy import accuracy
from src.lib.mapping import input_normalization_Matrix
from src.lib.backprop.gradient import gradient_descent_algorithm

# configuration file
# HLNN have structure [num_layer][num_neuron]
HLNN = pd.read_csv('../forward/HLNN.csv')
# OLNN have structure [num:output]
OLNN = pd.read_csv('../forward/OLNN.csv')

dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# Number of examples
l = 1000

X_D = dataset.iloc[:l, 1:]
#X_D = dataset.iloc[:, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:l, :1]
#Y_D = dataset.iloc[:, :1]
Y = Y_D[0].to_numpy()

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
# B is a list of vector
B = []

for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[0, 1], INPUT_DIMENSION))
    else:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[i, 1], HLNN.iloc[i - 1, 1]))
    W.append(w)

w = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, HLNN.iloc[HLNN.shape[0]-1, 1]))
W.append(w)

# B list of vectors
for i in range(HLNN.shape[0]):
    b = np.full((HLNN.iloc[i, 1], 1), -10.)
    B.append(b)

# only for computational reasons
b = np.full((OUTPUT_DIMENSION, 1), -10.)
B.append(b)

print('---------- Training ----------')

X = input_normalization_Matrix(X)

epochs = 200
learning_mode = 'batch'

W = gradient_descent_algorithm(Y, W, X, B, ETA, epochs)

#weight = pd.DataFrame(W)
# W-1L-batch-epochs

#file_name = 'W-BP-' + learning_mode + '-l=' + str(l) + '-epoch=' + str(epochs)

#weight.to_csv('weight-csv/' + file_name + '.csv', encoding='utf-8', header=False, index=False)

print('---------- Validation ----------')

# data 10000
validation_set = pd.read_csv('../../../data/mnist_test.csv', header=None)

# Number of examples
l = 5000
# parameters
XV_D = validation_set.iloc[:l, 1:]
#XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

YV_D = validation_set.iloc[:l, :1]
#YV_D = validation_set.iloc[:, :1]
YV = YV_D[0].to_numpy()

XV = input_normalization_Matrix(XV)

# file_name = 'W-1L-F-batch-l=5000-epoch=1000.csv'
# W = pd.read_csv('weight-csv/' + file_name, header=None)
# B = np.full(10, -10.)

print(accuracy(YV, W, XV, B), "%")