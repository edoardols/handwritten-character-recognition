import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

# configuration file
# HLNN have structure [num_layer][num_neuron]
HLNN =  pd.read_csv('../structure_neural_network/forward/HLNN.csv')
# OLNN have structure [num:output]
OLNN =  pd.read_csv('../structure_neural_network/forward/OLNN.csv')

# data
dataset = pd.read_csv('../../../data/mnist_train.csv', header=None)

# structor of the NN
dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# Number of examples
l = 100
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
B = []
W = []

# W list of matrices

np.random.seed(42)

for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[0,1],d))
    else:
        w = np.random.uniform(low=-1, high=1, size=(HLNN.iloc[i,1],HLNN.iloc[i-1,1]))
    W.append(w)

w = np.random.uniform(low=-1, high=1, size=(o,HLNN.iloc[HLNN.shape[0]-1,1]))
W.append(w)

# B list of vectors
for i in range(HLNN.shape[0]):
    b = np.full((HLNN.iloc[i,1],1), 10.)
    B.append(b)

# only for computational reasons
b = np.full((len(output_NN),1), 0.)
B.append(b)

def map_input(X):
    # input [0,255]
    # sigmoid [0,1]
    x = (0) + ((float(X) - 0)*(1 - (0)))/(255 - 0)
    return x

def map_label(y):
    # one hot encode
    Y = np.zeros(10, dtype = float)
    Y[y] = 1
    return Y

# activation

def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))

def output_layer(W,X,b):
    # W (KxN) is a matrix and X (Nx1) and b (Kx1) are vectors
    # K = #neurons layer i
    # N = #neurons layer i-1
    #A = np.full((W.shape[0],1), 0.)
    A = np.zeros((W.shape[0],1), dtype = float)
    A = np.dot(W,np.transpose(X)) + np.transpose(b.flatten())
    for i in range(0, len(A)):
        A[i] = sigmoid(A[i])
    return A

def output_NN(W,X,B):
    # W is a LIST of matrices , B is LIST of vector, X is a SINGLE input vector
    OUT_NN = []
    for i in range(len(W)): # loop for layers in ONE NN + OUTPUT NN
        w = W[i] # w is a matrices
        b = B[i] # B is a of vector
        if i == 0:
            X = X
        else:
            X = OUT_NN[i-1] # OUT layer before
        #print(w)
        OUT_NN.append(output_layer(w,X,b))
    return OUT_NN

def output_for_all_input(W,X,B):
    OUT_NN_ALL = [] # l is the index for a specific X input
    for i in range(X.shape[0]):
        x = X[i] # input vector
        OUT_NN_ALL.append(output_NN(W,x,B))
    return OUT_NN_ALL

def gradient_descent(W, eta, dfE):
    # W is a LIST of matrices
    for k in range(0,len(W)): 
        # W[k] is a k-th matrix in the W LIST
        for i in range(0,len(W[k])):
            # W[k][i] is a i-th row in the k-th matrix
            for j in range(0,len(W[k][i])):
                # W[k][i][j] is the i,j element in the k-th matrix
                W[k][i][j] = W[k][i][j] - eta*dfE
    return W

# TODO BACKPROP

def epochs_gradient_descent(epochs, X, B, Y, eta):
    global W
    for i in range(0,epochs):
        # calculate output NN
        out = output_for_all_input(W,X,B)
        #print(out)
        # list of output NN for every label
        Y_NN = []

        # TODO move in the output function
        for j in range(len(out)):
            # output of the NN for the single input
            y_nn = out[j][len(out[0])-1]
            Y_NN.append(y_nn[0])
    
        #dfemprisk = empirical_risk(Y,Y_NN)
        print('Empirical Risk step ' + str(i) + ' :'+ str(dfemprisk))
        W = gradient_descent(W, eta, dfemprisk)
        #print('---------- W ----------')
        #print(W)

def accuracy(Y,Y_NN):
    a = 0
    for i in range(0,len(Y)):
        # index of the max element in the array
        # what character the NN thiks it has recognized
        y_nn = np.argmax(Y_NN[i])
        y = Y[i]
        if y_nn == y:
            a = a + 1
    a = (a/len(Y))*100
    return a

print('---------- Training ----------')

#print(X[0])

# This function map pixel input from range [0,255] to range [0,1] it is a normalizzation
X_MAP = np.full((X.shape[0],X.shape[1]), 0.)
#print(X_MAP)
for i in range(0,X.shape[0]):
    for j in range(0,X.shape[1]):
        X_MAP[i][j] = map_input(X[i][j])

#print(X_MAP[0])

epochs_gradient_descent(10, X, B, Y, eta)

print('---------- Validation ----------')


# data
validation_set = pd.read_csv('../../../data/mnist_train.csv', header=None)

# structor of the NN
trainset_nrow = validation_set.shape[0]
trainset_ncolumn = validation_set.shape[1]

# Number of examples
l = 100
# parameters
XV_D = validation_set.iloc[:l, 1:]
XV = XV_D.to_numpy()

YV_D = validation_set.iloc[:l, :1]
YV = YV_D[0].to_numpy()

out = output_for_all_input(W,X,B)

# list of output NN for every label
YV_NN = []
for j in range(len(out)):
    # output of the NN for the single input
    y_nn = out[j][len(out[0])-1]
    YV_NN.append(y_nn[0])

acc = accuracy(YV,YV_NN)
print(acc)

# x = np.array(range(0, X.shape[0]))
# y = Y
# y_nn = Y_NN

# plt.title("Plotting Target")
# plt.xlabel("# Input")
# plt.ylabel("Target")

# plt.plot(x, y, color="red", marker="o", label="Y")
# plt.plot(x, y_nn, color="blue", marker="o", label="Y_NN")
# plt.legend()
# plt.show()