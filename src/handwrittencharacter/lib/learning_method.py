import numpy as np
import torch


def learning_method(D, method='mini', minibatch=128):

    # shuffle the dataset
    # np.random.shuffle(D)
    D = D[torch.randperm(D.size()[0])]

    X = D[:, 1:]
    Y = D[:, :1]

    x = []
    y = []
    if method == 'batch':
        y.append(Y)
        x.append(X)
        return y, x

    if method == 'mini':
        k = 0
        q = len(X) // minibatch
        r = len(X) % minibatch
        for i in range(0, q):
            x.append(X[i * minibatch:(i + 1) * minibatch, :])
            y.append(Y[i * minibatch:(i + 1) * minibatch])

        if r > 0:
            x.append(X[q * minibatch:, :])
            y.append(Y[q * minibatch:])
        return y, x

    if method == 'online':
        for i in range(0, len(X)):
            x.append(X[i])
            y.append(Y[i])
        return y, x
