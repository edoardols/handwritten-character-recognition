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
Y = Y.to_numpy()

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

# W list of matrices

for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.ones([HLNN.iloc[0,1],d], dtype = int)
    else:
        w = np.ones([HLNN.iloc[i,1],HLNN.iloc[i-1,1]], dtype = int)
    W.append(w)

w = np.ones([o,HLNN.iloc[HLNN.shape[0]-1,1]], dtype = int)
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
    for i in range(len(W)):
        
        #print(W)
        #print(X)
        a = a + W[i]*X[i]
        #a = 10
    #print(a)
    a = a + b
    
    #print(a)
    return a

# signmoid function
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def output_neuron(W,X,b):
    a = activation(W,X,b)
    sig = sigmoid(int(a))
    #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    #print(sig)
    #return 1
    return sig

def output_layer(W,X,B): # W is a matrices , B is a vector, X is a SINGLE input vector
    OUT_i = []
    for i in range(W.shape[0]): # loop for neurons in ONE layer #rows of the W
        # print('&&&&&&&&&&&&&&&&&&&&')
        # print(W.shape[0])
        # print(W)
        # print('-----')
        # print(B.shape[0])
        # print(B)
        w = W[i] # w is a vector
        b = B[i] # b is a scalar
        #print(b)
        out = output_neuron(w,X,b[0])
        #print(out)
        OUT_i.append(out) # X is a vector
        #OUT_i.append(0)
    #print('&&&&&&&&&&&&&&&&&&&&')
    #print(OUT_i)
    return OUT_i

def ouput_all_layer(W,X,B): # W is a LIST of matrices , B is LIST of vector, X is a SINGLE input vector
    OUT = []
    for i in range(len(W)): # loop for layers in ONE NN + OUTPUT NN
        w = W[i] # w is a matrices
        b = B[i] # B is a of vector
        #print('###############')
        #print(w)
        if i == 0:
            X = X
        else:
            X = OUT[i-1] # OUT layer before
            #print(OUT[i-1])
        OUT.append(output_layer(w,X,b))
    # print('###############')
    # print(OUT)
    return OUT

def outpul_for_all_input(W,X,B):
    OUT_l = [] # l is the index for a specific X input
    for i in range(X.shape[0]):
        x = X[i] # input vector
        OUT_l.append(ouput_all_layer(W,x,B))
    return OUT_l

# print(W) # LIST of matrices
# print('----')
# print(X) # ARRAY
# print('----')
# print(B) # LIST of vectors
# print('----')
# print(l)
# print('----------------------')
# print(X.shape)
#print(outpul_for_all_input(W,X,B))
out = outpul_for_all_input(W,X,B)

#print(out)
for i in range(len(out)):
    print('----------------------')
    print(out[0][len(out[0])-1])
    print(Y[i])

