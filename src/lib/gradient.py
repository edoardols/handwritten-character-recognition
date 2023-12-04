from src.lib.mean_square_error import loss_function

# TODO implement mode type for gradient descent: batch, online, mini-batch
def gradient_descent(Y, W, X, B, ETA, epochs=1):
    for i in range(0, epochs):
        E = loss_function(Y, W, X, B)
        W = W - ETA * E
    return W
