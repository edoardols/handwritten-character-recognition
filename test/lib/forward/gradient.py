from handwrittencharacter.lib import empirical_risk
from handwrittencharacter.lib import learning_method


# def gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode):
def gradient_descent_algorithm(D, WB, ETA, e, learning_mode):
    YB, XB = learning_method(D, learning_mode, 128)

    E_plot = 0
    for i in range(0, len(XB)):
        # Create E are list of matrix
        E, E_plot_batch = empirical_risk(YB[i], WB, XB[i])
        # gradient descent
        WB = WB - ETA * E

        E_plot = E_plot + E_plot_batch

        print('epoch: ', e+1, 'Batch: ', i+1)

    return WB, E_plot
