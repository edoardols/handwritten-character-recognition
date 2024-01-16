from handwrittencharacter.lib import learning_method


# def gradient_descent_algorithm(Y, W, X, B, ETA, e, learning_mode):
def gradient_descent_algorithm(dataset, W, B, ETA, e, learning_mode):
    # W is a list of matrices
    # B is list of vectors
    # YB, XB = learning_method(Y, X, learning_mode, 128)
    YB, XB = learning_method(dataset, learning_mode, 128)

    E_plot = 0
    for i in range(0, len(XB)):
        # Create E are list of matrix
        E, E_plot_batch = empirical_risk(YB[i], W, XB[i], B)

        for j in range(0, len(W)):
            W[j] = W[j] - ETA * E[j]

        E_plot = E_plot + E_plot_batch
        #E_epoch = E_epoch + np.sum(np.abs(E[len(E)-1]))

        print('epoch: ', e+1, 'Batch: ', i+1)
    return W, E_plot