from timeit import Timer

import numpy as np
arr = np.random.sample((60000, 28*28))
#print(arr)


def timeline_sample(series, num):
    random = series.copy()
    for i in range(num):
        np.random.shuffle(random)
        yield random
    # print(random)


def timeline_sample_fast(series, num):
    random = series.copy()
    for i in range(num):
        np.random.shuffle(random)
        yield random.T
    # print(random)


def timeline_sample_faster(series, num):
    length = arr.shape[0]
    for i in range(num):
        yield series[np.random.permutation(length), :]
        # yield series[:, np.random.permutation(length)]
        # print(series[np.random.permutation(length), :])


def consume(iterable):
    for s in iterable:
        np.sum(s)


print(min(Timer(lambda: consume(timeline_sample(arr, 1))).repeat(10, 10)))
print(min(Timer(lambda: consume(timeline_sample_fast(arr, 1))).repeat(10, 10)))
print(min(Timer(lambda: consume(timeline_sample_faster(arr, 1))).repeat(10, 10)))
#>>> 0.2585161680035526
#>>> 0.2416607110062614
#>>> 0.04835709399776533