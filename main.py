# define resoltuion of data (handwritten character pixel)
#resolution = (28,28)

# d = dimension of input (features) 
# l = number of input

# activation: a_i = sum_[j=1,d] ( w_ij * x_j) + b_i 
# neurons output: x_i = sigma(a_i)

# loss function: sum[y_i - sigma(a_i)]**2

# empirical risk: 1/2 * sum_[k=1,l](loss function)

import pandas as pd
import math

dataset = pd.read_csv('dataset\mnist_train.csv')

# get a row in the dataset specifying the index row
#print(csvfile.iloc[1][0]) # row into a column vector #get label

# structor of the NN

#print(csvfile.loc[0])
# d = 28*28 # 

weight = []

#print(dataset.shape[0])

dataset_nrow = dataset.shape[0]
dataset_ncolumn = dataset.shape[1] 

d = dataset_ncolumn - 1

# one hot-econding for output
output = [0] * 10
output_dim = len(output)
#print(output)

# 1 layer NN
weight_dim = d * output_dim
weight = [1] * weight_dim # TO  DO MAKE A MATRIX add DIM LAYER NN

# activation: a_i = sum_[j=1,d] ( w_ij * x_j) + b_i 
def activation(w,x,b):
    a = b
    for j in range(0,len(x)):
        a = a + w[j]*x[j]
    return a

# neurons output
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

# loss function: sum[y_i - sigma(a_i)]**2

def loss_function(y,w,x,b):
    sum = 0
    for i in len(y):
        y[i] - sigmoid(activation(w[i],x[i],b[i]))
    

# empirical risk: 1/2 * sum_[k=1,l](loss function)
