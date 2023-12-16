from src.lib.forward.mean_square_error import loss_function


def gradient_descent_algorithm(Y, W, X, B, ETA, epochs, iteration):
    for i in range(0, epochs):
        E = loss_function(Y, W, X, B)
        # gradient descent
        W = W - ETA * E
        print('Batch: ', iteration+1, ', epoch: ', i+1)
    return W
