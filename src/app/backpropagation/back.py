import pandas as pd
import numpy as np
import math

# configuration file
# HLNN have structure [num_layer][num_neuron]
HLNN = pd.read_csv('../forward/HLNN.csv')
# OLNN have structure [num:output]
OLNN = pd.read_csv('../forward/OLNN.csv')

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
o = OLNN.iloc[0, 1]
# initialization output
output_NN = [0] * o

# Layout Neural Network
np.random.seed(42)

# Layout Neural Network
# W is a list of matrix
W = []
# B is a list of vector
B = []

for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[0,1], d))
    else:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[i, 1], HLNN.iloc[i - 1, 1]))
    W.append(w)

w = np.random.uniform(low=-1, high=1, size=(o,HLNN.iloc[HLNN.shape[0]-1,1]))
W.append(w)

# B list of vectors
for i in range(HLNN.shape[0]):
    b = np.full((HLNN.iloc[i, 1], 1), 0.)
    B.append(b)

# only for computational reasons
b = np.full((len(output_NN),1), 0.)
B.append(b)

