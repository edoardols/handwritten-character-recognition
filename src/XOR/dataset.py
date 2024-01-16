
import numpy as np
import pandas as pd

# Your original matrix
xor = np.array([[0, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1]])

# Use numpy.tile to repeat the matrix and create a larger matrix
# result_matrix = np.tile(xor, (15000, 1))
# np.random.shuffle(result_matrix)
#
# # print("Original Matrix:")
# # print(xor)
# #
# # print("\nResult Matrix:")
# # print(result_matrix.shape)
#
# xor = pd.DataFrame(result_matrix)
# xor.to_csv('XOR_training.csv', encoding='utf-8', header=False, index=False)

result_matrix = np.tile(xor, (2500, 1))
np.random.shuffle(result_matrix)

xor = pd.DataFrame(result_matrix)
xor.to_csv('XOR_test.csv', encoding='utf-8', header=False, index=False)


