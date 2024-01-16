from src.XOR.lib.learning_method import learning_method
from src.XOR.lib.mean_square_error import empirical_risk


def gradient_descent_algorithm(dataset, WB0, WB1, ETA, e, learning_mode):
    YB, XB = learning_method(dataset, learning_mode, 128)

    E_plot = 0
    for i in range(0, len(XB)):
        E0, E1, E_plot_batch = empirical_risk(YB[i], WB0, WB1, XB[i])

        WB0 = WB0 - ETA * E0
        WB1 = WB1 - ETA * E1

        E_plot = E_plot + E_plot_batch

        print('epoch: ', e+1, 'Batch: ', i+1)
    return WB0, WB1, E_plot