import pandas as pd
import numpy as np

from src.lib.accuracy import accuracy
from src.lib.mapping import input_normalization_Matrix

print('---------- Validation ----------')

# data 10000
validation_set = pd.read_csv('../../../data/mnist_test.csv', header=None)

# Number of examples
l = 100
# parameters
XV_D = validation_set.iloc[:l, 1:]
# XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

# YV_D = validation_set.iloc[:, :1]
YV_D = validation_set.iloc[:l, :1]
YV = YV_D[0].to_numpy()

XV = input_normalization_Matrix(XV)

file_name = 'W-1L-F-batch-l=5000-epoch=1000.csv'
W = pd.read_csv('weight-csv/' + file_name, header=None)
B = np.full(10, -10.)

print(accuracy(YV, W, XV, B))
