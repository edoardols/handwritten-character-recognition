import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

# configuration file
# OLNN have structure [num:output]
OLNN =  pd.read_csv('../neural/OLNN.csv')

# dataset
dataset = pd.read_csv('../dataset/mnist_train.csv', header=None)

# structor of the NN
dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# Number of examples
l = 1000
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
#Y_NN = np.zeros((o), dtype = float)

# Layout Neural Network
np.random.seed(42)

# W is a matrix
W = np.random.uniform(low=-10, high=10, size=(o,d))

# B Vector
B = np.full(o, -10.)

############### FUNCTIONS ################

def map_input(X):
    # input [0,255]
    # sigmoid [0,1]
    x = (0) + ((float(X) - 0)*(1 - (0)))/(255 - 0)
    return x

def map_label(y):
    # one hot encoding
    Y = np.zeros((10,1), dtype = float)
    Y[y] = 1
    return Y

def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))

def dsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

sigMatrix = np.vectorize(sigmoid)
dsigMatrix = np.vectorize(dsigmoid)

def activation(W,X,B):
    # W dxO is a matrix and X is a dx1 and B is a Ox1 vectors
    # O = #neurons in output
    # d = #neurons in input
    A = np.zeros((len(W)), dtype = float)
    A = np.dot(W,np.transpose(X)) + B
    return A

# def output_NN(W,X,B):
#     # W (KxN) is a matrix and X (Nx1) and b (Kx1) are vectors
#     # K = #neurons layer i
#     # N = #neurons layer i-1
#     A = np.zeros((len(W)), dtype = float)
#     A = np.dot(W,np.transpose(X)) + B
#     return sigMatrix(A)

def gradient_descent(W, X, eta, dfE):
    # TODO
    # W = W - eta*X*derror
    # W is oxd
    # X is 1xd
    print('dfe ', dfE.shape)
    print('X', X.shape)
    print('W', W.shape)
    #print(W)
    for i in range(0,len(X)):
        #W = W - eta*np.dot(np.transpose(X[i]),dfE[i])
        #print(dfE)
        d_dot = dfE[i].reshape(-1,1)
        x_dot = X[i].reshape(1, -1)
        Z = np.dot(d_dot, x_dot)
        #print(np.size(Z))
        W = W - eta*Z
    #print(W)
    # print(dfE)
    # print('------')
    # print(W)
    # for i in range(0,len(W)):
    #     # W[i] is a i-th row in the W matrix
    #     for j in range(0,len(W[i])):
    #         # W[i][j] is the i,j element in the W matrix
    #         #print(W[i][j])
    #         W[i][j] = W[i][j] - eta*dfE
    return W

def delta_error(Y, W, X, B):
    A = activation(W,X,B)
    E = (Y - sigMatrix(A) )*dsigMatrix(A)
    d_error = np.dot(E,X)
    #print('---')
    #print(sigMatrix(A))
    #print(np.sum(d_error))
    print('--- derror')
    print(d_error)
    return d_error
    #print(d_error)
    return np.sum(d_error)

# TEST
# X_MAP = np.full((X.shape[0],X.shape[1]), 0.)
# #print(X_MAP)
# for i in range(0,X.shape[0]):
#     for j in range(0,X.shape[1]):
#         X_MAP[i][j] = map_input(X[i][j])

# test = output_NN(W,X_MAP[0],B)

# o = delta_error(Y, W, X_MAP[0], B)

# print(o)




def epochs_gradient_descent(epochs, X, B, Y, eta):
    global W
    for j in range(0,epochs):
        de = 0
        for i in range(0,len(X)):
            y = map_label(Y[i])
            de = de + delta_error(y, W, X[i], B)
            W = gradient_descent(W, X, eta, de)
        #print('Empirical Risk step ' + str(j) + ' :'+ str(de))
        
        # print('---------- W ----------')
        # print(W)

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

epochs_gradient_descent(50, X_MAP, B, Y, eta)

print('---------- Validation ----------')


# dataset
validation_set = pd.read_csv('../dataset/mnist_train.csv', header=None)

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

XV_MAP = np.full((XV.shape[0],XV.shape[1]), 0.)
#print(X_MAP)
for i in range(0,XV.shape[0]):
    for j in range(0,XV.shape[1]):
        XV_MAP[i][j] = map_input(XV[i][j])

# list of output NN for every label
YV_NN = []
for i in range(len(XV)):
    # output of the NN for the single input
    y_nn = sigMatrix(activation(W,XV_MAP[i],B))
    YV_NN.append(y_nn)

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