from src.handwrittencharacter.lib.learning_method import learning_method
from src.handwrittencharacter.lib.backprop.mean_square_error import empirical_risk

import pandas as pd

import os

def gradient_descent_algorithm(dataset, WB0, WB1, WB2, ETA, e, learning_mode, batch_dimension):

    YB, XB = learning_method(dataset, learning_mode, batch_dimension)

    E_plot = 0
    for i in range(0, len(XB)):
        E0, E1, E2, E_plot_batch = empirical_risk(YB[i], WB0, WB1, WB2, XB[i])

        WB0 = WB0 - ETA * E0
        WB1 = WB1 - ETA * E1
        WB2 = WB2 - ETA * E2

        E_plot = E_plot + E_plot_batch

        print('epoch: ', e+1, 'Batch: ', i+1)
    return WB0, WB1, WB2, E_plot