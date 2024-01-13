import numpy as np

# input dim
INPUT_DIMENSION = 28 * 28

# output dim
OUTPUT_DIMENSION = 10

print('X: (',str(INPUT_DIMENSION), ', 1)')
# W is a matrix
W = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, INPUT_DIMENSION))
print('W:', W.shape)

# B is a vector
B = np.random.uniform(low=-1, high=1, size=(OUTPUT_DIMENSION, 1))
print('B:', B.shape)
