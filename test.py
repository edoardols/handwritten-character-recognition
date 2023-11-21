

# --------
def dsigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

# loss function: sum[y_i - sigma(a_i)]**2

# def least_square(y,w,x,b):
#     sum = 0
#     for i in len(y):
#         a = activation(w[i],x[i],b[i])
#         error = y[i] - sigmoid(a)
#         sum = sum + error**2
#     return sum
    
# # empirical risk: 1/2 * sum_[k=1,l](loss function)
# def empirical_risk(y,w,x,b,l):
#     sum = 0
#     for i in range(0,l):
#         sum = sum + least_square(y,w,x,b)
#     return 1/2 * sum

def delta_error(y,W,X,b):
    a = activation(W,X,b)
    return (y - sigmoid(a) ) * ( - dsigmoid(a))

def gradient(y,w,x,b,l):
    sum = 0
    for i in range(0,l):
        sum = sum + delta_error(y[i],w[i],x[i],b[i]) * x[i]
    return sum

# TODO recursive function

# def output(W,X,B):
#     # W and X is a matrix
#     Y = []
#     for i in range(W.shape[0]):
#         # W[i] and X[i] are vectors of all connection for 1 neuron
#         a = activation(W[i],X[i],B[i])
#         # function
#         y = sigmoid(a)
#         Y.append(y)
#     return Y
    

def gradient_descent(Y,W,X,B,l,eta):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            # if i = 0 X = input
            # if i != 0 X = Y output step i-1
            W[i][j] = W[i][j] - (eta * gradient(Y,W[i][j],X,B[i][j],l))

    #return W
    #print(W)


#print(w)


#print(gradient_descent(y,w,x,b,l,eta))
#print(dataset.columns[0])

#print(W)
# for i in range(len(W)):
#     print(W[i])
#     print('---')

# for i in range(len(B)):
#     print(B[i])
#     print('---')
Y_layer = []

def output(W,X,B):
    global Y_layer
    for i in range(W.shape[0]):
        # W[i] and X[i] are vectors of all connection for 1 neuron
        a = activation(W[i],X[i],B[i])
        # function
        y = sigmoid(a)
        Y.append(y)


def grad_layer(Y_,W_,X_,B_,l,eta):
    for i in range(len(W)):
        W = W_[i]
        B = B_[i]
        X = X_
        #gradient_descent(Y,w,X,b,l,eta)
        print(w)
        print(b)
        if i == 0:
            y_later = X_
        y_later = Y[i-1]

        if i != 0:
            Y = output(W,X,B)



grad_layer(Y,W,X,B,l,eta)