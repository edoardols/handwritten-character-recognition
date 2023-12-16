def learning_method(label, dataset, method='mini-batch', minibatch=16):
    x = []
    y = []
    if method == 'batch':
        return y.append(label), x.append(dataset)

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
