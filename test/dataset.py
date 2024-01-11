import pandas as pd
import numpy as np

F = np.array([[0, 0, 0],
         [1, 0, 1],
         [1, 1, 0],
         [0, 1, 1]])

B = []
for i in range(0, 10000):
    x = np.random.randint(4)
    #print(x)
    #print(F[x])
    B.append(F[x])

patter = pd.DataFrame(B)
patter.to_csv('XOR_val.csv', encoding='utf-8', header=False, index=False)
