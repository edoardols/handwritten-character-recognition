def learning_method(dataset, method='mini-batch', minibatch=10):
    x = []
    if method == 'batch':
        return x.append(dataset)

    if method == 'mini':
        k = 0
        q = len(dataset) // minibatch
        r = len(dataset) % minibatch
        for i in range(0, q):
            x.append(dataset[i * minibatch:(i + 1) * minibatch, :])

        if r > 0:
            x.append(dataset[q * minibatch:, :])
        return x

    if method == 'online':
        for i in range(0, len(dataset)):
            x.append(dataset[i])
        return x
