import pandas as pd
import numpy as np
import math

# configuration file
# OLNN have structure [num:output]
OLNN = pd.read_csv('../structure_neural_network/forward/OLNN.csv')

# data
dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# structor of the NN
dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# Number of examples
l = 5000
# parameters
X_D = dataset.iloc[:l, 1:]
X = X_D.to_numpy()

Y_D = dataset.iloc[:l, :1]
Y = Y_D[0].to_numpy()

# dim input
d = dataset_ncolumn - 1

# learning rate
eta = 0.01

# output dim
o = OLNN.iloc[0,1]
# initialization output
output_NN = [0] * o

# Layout Neural Network
np.random.seed(42)

# W is a matrix
W = np.random.uniform(low=-1, high=1, size=(o,d))

wdf = pd.DataFrame(W)

wdf.to_csv("weight-pre.csv", encoding='utf-8',header=False, index=False)