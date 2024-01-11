import numpy as np

from lib_test.mapping import input_normalization_Matrix


def learning_method(D, method='mini', minibatch=16):

    # shuffle the dataset
    #dataset.sample(frac=1)
    np.random.shuffle(D)

    # X_D = dataset.iloc[:, 1:]
    # X = X_D.to_numpy()
    #
    # X = input_normalization_Matrix(X)
    #
    # Y_D = dataset.iloc[:, :1]
    # Y = Y_D[0].to_numpy()

    X = D[:, 1:]
    Y = D[:, :1]

    dataset = X
    label = Y

    x = []
    y = []
    if method == 'batch':
        y.append(label)
        x.append(dataset)
        return y, x

    if method == 'mini':
        k = 0
        q = len(dataset) // minibatch
        r = len(dataset) % minibatch
        for i in range(0, q):
            x.append(dataset[i * minibatch:(i + 1) * minibatch, :])
            y.append(label[i * minibatch:(i + 1) * minibatch])

        if r > 0:
            x.append(dataset[q * minibatch:, :])
            y.append(label[q * minibatch:])
        return y, x

    if method == 'online':
        for i in range(0, len(dataset)):
            x.append(dataset[i])
            y.append(label[i])
        return y, x
