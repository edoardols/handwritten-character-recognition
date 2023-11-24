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

weight = []

#print(dataset.shape[0])

dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1]

# dumber of examples
#l = dataset.shape[0]
l = 1000
# parameters
X = dataset.iloc[:l, 1:]
X = X.to_numpy()

Y = dataset.iloc[:l, :1]
Y = Y.to_numpy()

# dim input
d = dataset_ncolumn - 1

# learning rate
eta = 0.01

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

norm_factor = float(1/255)

for i in range(HLNN.shape[0]):
    if i == 0:
        w = np.ones([HLNN.iloc[0,1],d], dtype = float)
    else:
        w = np.ones([HLNN.iloc[i,1],HLNN.iloc[i-1,1]], dtype = float)
    # w = w*float(0.5) provides a factorizzation
    w = w*norm_factor
    W.append(w)

w = np.ones([o,HLNN.iloc[HLNN.shape[0]-1,1]], dtype = float)
w = w*norm_factor
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

def activation(W,X,b): # W and X are vectors, b is scalar
    a = 0
    for i in range(len(W)):
        a = a + W[i]*X[i]
    a = a + b
    
    #print(a)
    return a

# signmoid function
def sigmoid(a):
    if a < -10:
        return 0.000045
    if a > 10:
        return 0.999955
    return 1 / (1 + math.exp(-a))

def output_neuron_NN(W,X,b):
    a = activation(W,X,b)
    sig = sigmoid(int(a))
    return sig

def calculation_output_NN(W,X,b):
    # we dont apply the sigmoid transformation
    a = activation(W,X,b)
    return a

def output_layer(W,X,B): # W is a matrices , B is a vector, X is a SINGLE input vector
    OUT_i = []
    for i in range(W.shape[0]): # loop for neurons in ONE layer #rows of the W
        w = W[i] # w is a vector
        b = B[i] # b is a scalar
        if (i == W.shape[0] - 1):
            out = calculation_output_NN(w,X,b[0])
        else:
                    out = output_neuron_NN(w,X,b[0])
        OUT_i.append(out) # X is a vector
    return OUT_i

def ouput_all_layer(W,X,B): # W is a LIST of matrices , B is LIST of vector, X is a SINGLE input vector
    OUT = []
    for i in range(len(W)): # loop for layers in ONE NN + OUTPUT NN
        w = W[i] # w is a matrices
        b = B[i] # B is a of vector
        if i == 0:
            X = X
        else:
            X = OUT[i-1] # OUT layer before
        OUT.append(output_layer(w,X,b))
    return OUT

def output_for_all_input(W,X,B):
    OUT_l = [] # l is the index for a specific X input
    for i in range(X.shape[0]):
        x = X[i] # input vector
        OUT_l.append(ouput_all_layer(W,x,B))
    return OUT_l




def dsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

def insigmoid(siga):
    # np.log = natural log
    return np.log(siga / (1 - siga) )

def delta_error(y,y_nn):  # Y is the target, Y_NN in the output of the NN for a that speific input
    a = y_nn
    y_comparable = sigmoid(y)
    error = (y_comparable - sigmoid(a) ) * ( - dsigmoid(a))
    return error

def dfempirical_risk(X,Y, Y_NN):
    sum = 0
    for i in range(0,X.shape[0]):
        sum = sum + (delta_error(Y[i], Y_NN[i]))**2
    return sum/2

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


    # print('Output: ' + str(y_nn[0]))
    # y = Y[i]
    # print('Character: ' + str(y[0]))
    # fy_nn = float(str(y_nn[0]))
    # fy = float(str(y[0]))
    # derror = delta_error(fy,fy_nn)
    # print('Delta Error: ' + str(derror))


def epochs_gradient_descent(epochs, X, B, Y, eta):
    global W
    for i in range(0,epochs):
        # calculate output NN
        out = output_for_all_input(W,X,B)
        #print(out[1])
        # list of output NN for every label
        Y_NN = []
        for j in range(len(out)):
            # output of the NN for the single input
            y_nn = out[j][len(out[0])-1]
            Y_NN.append(y_nn[0])
    
        dfemprisk = dfempirical_risk(X,Y,Y_NN)
        print('Empirical Risk step ' + str(i) + ' :'+ str(dfemprisk))
        W = gradient_descent(W, eta, dfemprisk)
        #print(W)


epochs_gradient_descent(10, X, B, Y, eta)

out = output_for_all_input(W,X,B)
# list of output NN for every label
Y_NN = []
for j in range(len(out)):
    # output of the NN for the single input
    y_nn = out[j][len(out[0])-1]
    Y_NN.append(y_nn[0])
    print(y_nn)
    print(Y[j])