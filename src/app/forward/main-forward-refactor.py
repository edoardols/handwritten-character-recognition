import pandas as pd
import numpy as np

from src.lib.mapping import input_normalization_Matrix
from src.lib.gradient import gradient_descent

# configuration file
# OLNN have structure [num:output]
OLNN = pd.read_csv('OLNN.csv')

# data 60000
dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# structure of the NN
dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# Number of examples
l = 5000
# parameters
X_D = dataset.iloc[:l, 1:]
#X_D = dataset.iloc[:, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:l, :1]
#Y_D = dataset.iloc[:, :1]
Y = Y_D[0].to_numpy()

# dim input
#d = dataset_ncolumn - 1
d = len(X)

# learning rate
eta = 0.01

# output dim
o = OLNN.iloc[0, 1]
# initialization output
output_NN = [0] * o

# Layout Neural Network
np.random.seed(42)

# W is a matrix
# W = np.random.uniform(low=-1, high=1, size=(o,d))

W = pd.read_csv('weight-pre.csv', header=None)

# B Vector
B = np.full(o, -10.)

def epochs_gradient_descent(epochs, Y, X, B, eta):
    global W
    for i in range(0, epochs):
        print('epoch: ', i)
        W = gradient_descent(Y, W, X, B, eta)


print('---------- Training ----------')

# This function map pixel input from range [0,255] to range [0,1] it is a normalization
#X_MAP = np.full((X.shape[0], X.shape[1]), 0.)
# print(X_MAP)
#for i in range(0, X.shape[0]):
#    for j in range(0, X.shape[1]):
#        X_MAP[i][j] = input_normalization(X[i][j])

X = input_normalization_Matrix(X)

epochs = 1000
epochs_gradient_descent(epochs, Y, X, B, eta)

wdf = pd.DataFrame(W)

wdf.to_csv("weight-post-" + str(epochs) + ".csv", encoding='utf-8', header=False, index=False)

print('---------- Validation ----------')

# data 10000
validation_set = pd.read_csv('../../../data/mnist_test.csv', header=None)

# structure of the NN
trainset_nrow = validation_set.shape[0]
trainset_ncolumn = validation_set.shape[1]

# Number of examples
l = 100
# parameters
# XV_D = validation_set.iloc[:l, 1:]
XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

YV_D = validation_set.iloc[:, :1]
# YV_D = validation_set.iloc[:l, :1]
YV = YV_D[0].to_numpy()

XV_MAP = np.full((XV.shape[0], XV.shape[1]), 0.)
for i in range(0, XV.shape[0]):
    for j in range(0, XV.shape[1]):
        XV_MAP[i][j] = input_normalization(XV[i][j])


# list of output NN for every label

def accuracy(Y, W, X, B):
    a = 0
    for i in range(0, len(X)):
        A = activation(W, X[i], B)
        Y_NN = sigMatrix(A)
        # index of the max element in the array
        # what character the NN thinks it has recognized
        y_nn = np.argmax(Y_NN)
        if y_nn == Y[i]:
            a = a + 1
        a = (a / len(Y)) * 100
    return a


acc = accuracy(YV, W, XV_MAP, B)
print(acc)
