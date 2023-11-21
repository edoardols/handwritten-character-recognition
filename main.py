# define resoltuion of data (handwritten character pixel)
#resolution = (28,28)

# d = dimension of input (features) 
# l = number of input

# activation: a_i = sum_[j=1,d] ( w_ij * x_j) + b_i 
# neurons output: x_i = sigma(a_i)

# loss function: sum[y_i - sigma(a_i)]**2

# empirical risk: 1/2 * sum_[k=1,l](loss function)

import pandas as pd
import numpy as np
import math

# configuration file
# HLNN have structure [num_layer][num_neuron]
HLNN =  pd.read_csv('neural\HLNN.csv')
# OLNN have structure [num:output]
OLNN =  pd.read_csv('neural\OLNN.csv')

# dataset
dataset = pd.read_csv('dataset\mnist_train.csv', header=None)

#print(dataset)

# get a row in the dataset specifying the index row
#print(csvfile.iloc[1][0]) # row into a column vector #get label

# structor of the NN

#print(csvfile.loc[0])
# d = 28*28 # 

weight = []

#print(dataset.shape[0])

dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# dumber of examples
#l = dataset.shape[0]
l = 10
# parameters
X = dataset.iloc[:l, 1:]
X = X.to_numpy()

Y = dataset.iloc[:l, :1]

# dim input
d = dataset_ncolumn - 1

# learning rate
eta = 0.1

# bias
B = []

# output dim
o = 1
# initialization output
output_NN = [0] * o

#print(output)

# Layout Neural Network
W = []
#print(HLNN.iloc[2, 1]) # layer 3 num neuroni

# TODO list range to fix 
#print(HLNN.shape)
# for i in range(HLNN.shape[0]):
#     if i == 0:
#         w = np.ones([d,HLNN.iloc[i,1]], dtype = int)
#     elif i == HLNN.shape[0] - 1:
#         w = np.ones([HLNN.iloc[i,1],o], dtype = int)
#     else:
#         w = np.ones([HLNN.iloc[i-1,1],HLNN.iloc[i,1]], dtype = int)
#     W.append(w)
#print(HLNN.shape)
# i = - 1

# ERROR COLUMNS ROWS INVERSION
# w = np.ones([d,HLNN.iloc[0,1]], dtype = int)
# W.append(w)

# #b = np.zeros([d,HLNN.iloc[0,1]], dtype = int)
# b = np.zeros([d,1], dtype = int)
# B.append(b)
# for i in range(HLNN.shape[0]):  
#     if i == HLNN.shape[0] - 1:
#         w = np.ones([HLNN.iloc[i,1],o], dtype = int)
#         # b = 0
#         #b = np.zeros([HLNN.iloc[i,1],o], dtype = int)
#         b = np.zeros([HLNN.iloc[i,1],1], dtype = int)
#     else:
#         w = np.ones([HLNN.iloc[i,1],HLNN.iloc[i+1,1]], dtype = int)
#         # b = 0
#         #b = np.zeros([HLNN.iloc[i,1],HLNN.iloc[i+1,1]], dtype = int)
#         b = np.zeros([HLNN.iloc[i,1],1], dtype = int)
#     B.append(b)
#     W.append(w)

# W list of matrices

w = np.ones([HLNN.iloc[0,1],d], dtype = int)
W.append(w)
for i in range(HLNN.shape[0]):  
    if i == HLNN.shape[0] - 1:
        w = np.ones([HLNN.iloc[i,1],o], dtype = int)
    else:
        w = np.ones([HLNN.iloc[i,1],HLNN.iloc[i+1,1]], dtype = int)
    W.append(w)

# B list of vectors

for i in range(HLNN.shape[0]):
    b = np.zeros([HLNN.iloc[i,1],1], dtype = int)
    B.append(b)

# only for computational reasons
b = np.zeros([len(output_NN),1], dtype = int)
B.append(b)

# STEP 1 # fixed 1 input

# calcolo OUTPUT (Errato) NN
    # calcolo da layer 0 a layer n degli output OUTPUT
    # => OUPUT = [OUT_1, ..., OUT_n]


# activation: a_i = sum_[j=1,d] ( w_ij * x_j) + b_i 

# ----- OUTPUT ------
def activation(W,X,b): # W and X are vectors, b is scalar
    a = 0
    # print(W)
    # print(X)
    # print(b)
    for j in range(len(W)):
        a = a + W[j]*X[j]
    a = a + b
    return a

# signmoid function
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def output_neuron(W,X,b):
    a = activation(W,X,b)
    #return sigmoid(int(a))
    return 1


OUT_i = []
def output_layer(W,X,B): #
    for i in range(len(W)): # loop for neurons in ONE layer
        w = W[i]
        b = B[i]
        #print(b)
        OUT_i.append(output_neuron(w,X,b))
        #OUT_i.append(0)
    return OUT_i

def ouput_all_layer(W,X,B): # W is a LIST of matrices , B is LIST of vector X is a SINGLE input vector
    OUT = []
    for i in range(len(W)): # loop for layers in ONE NN
        w = W[i]
        if i == 0:
            Y = X
        else:
            Y = OUT_i[i-1]
        OUT.append(output_layer(w,Y,B))
    return OUT

def outpul_for_all_input(W,X,B,l):
    OUT_l = [] # l is the index for a specific X input
    for i in range(l):
        x = X[i]
        #print(x)
        OUT_l.append(ouput_all_layer(W,x,B))
    return OUT_l

#print(W)
#print(len(W))
# print('----')
# print(X)
# print('----')
#print(B)
#print(len(B))
# print('----')
# print(l)
print(outpul_for_all_input(W,X,B,1))