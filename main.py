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

# parameters
#x =
#y =

#print(dataset.iloc[:10, :1])
#print(dataset.iloc[:10, 1:2])

l = dataset_nrow
d = dataset_ncolumn - 1
eta = 0.1 # learning rate

# output
output_dim = 1
output = [0] * output_dim

#print(output)

# x layer NN
num_layer = 3 # number of layer
num_neurons_for_layer  = 3
w = np.ones([num_neurons_for_layer, num_layer], dtype = int) # TO  DO MAKE A MATRIX add DIM LAYER NN

# activation: a_i = sum_[j=1,d] ( w_ij * x_j) + b_i 
def activation(w,x,b):
    a = b
    for j in range(0,len(x)):
        a = a + w[j]*x[j]
    return a

# signmoid function
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

def dsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

# loss function: sum[y_i - sigma(a_i)]**2

def least_square(y,w,x,b):
    sum = 0
    for i in len(y):
        a = activation(w[i],x[i],b[i])
        error = y[i] - sigmoid(a)
        sum = sum + error**2
    return sum
    
# empirical risk: 1/2 * sum_[k=1,l](loss function)
def empirical_risk(y,w,x,b,l):
    sum = 0
    for i in range(0,l):
        sum = sum + least_square(y,w,x,b)
    return 1/2 * sum

def delta_error(y,w,x,b):
    a = activation(w,x,b)
    return (y - sigmoid(a) ) * ( - dsigmoid(a))

def gradient(y,w,x,b,l):
    sum = 0
    for i in range(0,l):
        sum = sum + delta_error(y[i],w[i],x[i],b[i]) * x[i]
    return sum

def gradient_descent(y,w,x,b,l,eta):
    w = w - (eta * gradient(y,w,x,b,l))
    return w


#print(w)


#print(gradient_descent(y,w,x,b,l,eta))
#print(dataset.columns[0])

