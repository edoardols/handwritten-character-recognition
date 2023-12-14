from src.lib.forward.mean_square_error import loss_function


def gradient_descent_algorithm(Y, W, X, B, ETA, epochs=1):
    for i in range(0, epochs):
        E = loss_function(Y, W, X, B)
        # gradient descent
        W = W - ETA * E
        print('epoch: ', i)
    return W
