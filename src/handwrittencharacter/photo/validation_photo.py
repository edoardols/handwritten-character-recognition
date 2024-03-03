import numpy as np
import pandas as pd

from src.handwrittencharacter.lib.mapping import input_normalization_Matrix
from src.handwrittencharacter.lib.backprop.accuracy import output
from src.handwrittencharacter.lib.sigmoid import sigMatrix
from src.handwrittencharacter.lib.activation import activation


def forward_photo_validation(PATH_MAIN_FILE, weight_and_biases, digit):

    weight_and_biases_path = PATH_MAIN_FILE + '/../forward/training/' + weight_and_biases

    X = input_normalization_Matrix(digit)

    w = pd.read_csv(weight_and_biases_path + 'W.csv', header=None)
    W = w.to_numpy()

    b = pd.read_csv(weight_and_biases_path + 'B.csv', header=None)
    B = b.to_numpy()

    # Bias learnable
    # Expand matrix W with a column B
    WB = np.insert(W, W.shape[1], np.transpose(B), axis=1)

    # Expand matrix X with a column of 1s
    X_hat = np.insert(X, X.shape[1], np.transpose(np.ones((X.shape[0], 1), dtype=float)), axis=1)

    # Evaluate output of the Forward Neural Network
    A = activation(WB, X_hat)
    Y_NN = sigMatrix(A)

    # index of the max element in the array
    # what character the NN thinks it has recognized
    y_nn = np.argmax(Y_NN)

    return y_nn


def backpropagation_photo_validation(PATH_MAIN_FILE, weight_and_biases, digit):

    weight_and_biases_path = PATH_MAIN_FILE + '/../backpropagation/training/' + weight_and_biases

    X = input_normalization_Matrix(digit)

    # Weights
    w = pd.read_csv(weight_and_biases_path + 'W0.csv', header=None)
    W0 = w.to_numpy()
    w = pd.read_csv(weight_and_biases_path + 'W1.csv', header=None)
    W1 = w.to_numpy()
    w = pd.read_csv(weight_and_biases_path + 'W2.csv', header=None)
    W2 = w.to_numpy()

    # Biases
    b = pd.read_csv(weight_and_biases_path + 'B0.csv', header=None)
    B0 = b.to_numpy()
    b = pd.read_csv(weight_and_biases_path + 'B1.csv', header=None)
    B1 = b.to_numpy()
    b = pd.read_csv(weight_and_biases_path + 'B2.csv', header=None)
    B2 = b.to_numpy()

    # Expand matrix W with a column B
    WB0 = np.insert(W0, W0.shape[1], np.transpose(B0), axis=1)
    WB1 = np.insert(W1, W1.shape[1], np.transpose(B1), axis=1)
    WB2 = np.insert(W2, W2.shape[1], np.transpose(B2), axis=1)

    # Expand matrix X with a column of 1s
    X_hat = np.insert(X, X.shape[1], np.transpose(np.ones((X.shape[0], 1), dtype=float)), axis=1)

    # Evaluate output of the Backpropagation Neural Network
    Y_NN = output(WB0, WB1, WB2, X_hat[0])
    # index of the max element in the array
    # what character the NN thinks it has recognized
    y_nn = np.argmax(Y_NN)

    return y_nn
