from timeit import Timer

import numpy as np

X = np.random.uniform(low=0.0, high=1.0, size=(1, 28*28))
W = np.random.uniform(low=-1.0, high=1.0, size=(16, 28*28))

# print(X)
# print(W)
# print(np.dot(W, X.T))
# print(W.dot(X.T))


def np_dot(W, X, num):
    for i in range(num):
        yield np.dot(W, X.T)


def array_dot(W, X, num):
    for i in range(num):
        yield W.dot(X.T)

def activation(WB, X, num):
    for i in range(num):
        A = np.dot(WB, X.T)
        yield A


def transpose(X, num):
    for i in range(num):
        x = X.T
        yield x


def reshape(X, num):
    for i in range(num):
        x = X.reshape(-1, 1)
        yield x


def consume(iterable):
    for s in iterable:
        np.sum(s)


# print(min(Timer(lambda: consume(np_dot(W, X, 1))).repeat(60000, 1)))
# print(min(Timer(lambda: consume(array_dot(W, X, 1))).repeat(60000, 1)))
# print(min(Timer(lambda: consume(activation(W, X, 1))).repeat(60000, 1)))

# print(min(Timer(lambda: consume(transpose(X, 1))).repeat(60000, 1)))
# print(min(Timer(lambda: consume(reshape(X, 1))).repeat(60000, 1)))
a = 2.063274


def multiply(W, a, num):
    for i in range(num):
        w = W * a
        yield w


def mp_multiply(W, a, num):
    for i in range(num):
        w = np.multiply(W, a)
        yield w


# print(min(Timer(lambda: consume(multiply(W, a, 1))).repeat(60000, 1)))
# print(min(Timer(lambda: consume(mp_multiply(W, a, 1))).repeat(60000, 1)))


W1 = np.random.uniform(low=-1.0, high=1.0, size=(16, 28*28))
W2 = np.random.uniform(low=-1.0, high=1.0, size=(16, 28*28))


def numpy_sub(W1, W2, num):
    for i in range(num):
        x = np.subtract(W1, W2)
        yield x


def classic_sub(W1, W2, num):
    for i in range(num):
        x = W1 - W2
        yield x


print(min(Timer(lambda: consume(numpy_sub(W1, W2, 1))).repeat(60000, 1)))
print(min(Timer(lambda: consume(classic_sub(W1, W2, 1))).repeat(60000, 1)))