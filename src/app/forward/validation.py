import pandas as pd
import numpy as np

from src.lib.forward.accuracy import accuracy
from src.lib.mapping import input_normalization_Matrix

print('---------- Validation ----------')

# data 10000
dataset = 'mnist_test'
# dataset = 'salt_pepper/mnist_test-sp-s-0.3-p-0.2'
# dataset = 'salt_pepper/mnist_test-sp-s-0.5-p-0.6'
# dataset = 'salt_pepper/mnist_test-sp-s-1-p-0.6'

# dataset = 'blob/mnist_test-bl-p-0.2'

# dataset = 'thickness/mnist_test-th-step=-3'
validation_set = pd.read_csv('../../../data/' + dataset + '.csv', header=None)

# Number of examples
# parameters
XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

YV_D = validation_set.iloc[:, :1]
YV = YV_D[0].to_numpy()

XV = input_normalization_Matrix(XV)

# XV = np.transpose(XV)

# XV = XV.reshape(-1, 1)

file_name = 'W-1L-F-mini-l=10000-epoch=1000'
#file_name = 'W-1L-F-batch-l=5000-epoch=1000'
w = pd.read_csv('weight-csv/' + file_name + '.csv', header=None)
W = w.to_numpy()
B = np.full((10, 1), -10.)

print(accuracy(YV, W, XV, B), "%")
