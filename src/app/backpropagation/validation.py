import pandas as pd

from src.lib.backprop.accuracy import accuracy
from src.lib.mapping import input_normalization_Matrix

print('---------- Validation ----------')

# dataset = 'mnist_test'
# dataset = 'salt_pepper/mnist_test-sp-s-0.3-p-0.2'
# dataset = 'salt_pepper/mnist_test-sp-s-0.5-p-0.6'
# dataset = 'salt_pepper/mnist_test-sp-s-1-p-0.6'
dataset = 'blob/mnist_test-bl-p-0.2'
validation_set = pd.read_csv('../../../data/' + dataset + '.csv', header=None)

# parameters
XV_D = validation_set.iloc[:, 1:]
XV = XV_D.to_numpy()

YV_D = validation_set.iloc[:, :1]
YV = YV_D[0].to_numpy()

XV = input_normalization_Matrix(XV)


file_name = 'W-BP-mini-l=60000-epoch=5'

W = []
B = []

for i in range(0, 3):
    b = pd.read_csv('weight-csv/' + file_name + '/' + 'B' + str(i) + '.csv', header=None)
    b = b[0].to_numpy()
    B.append(b.reshape(-1, 1))

    w = pd.read_csv('weight-csv/' + file_name + '/' + 'W' + str(i) + '.csv', header=None)
    W.append(w.to_numpy())

print(accuracy(YV, W, XV, B), "%")
