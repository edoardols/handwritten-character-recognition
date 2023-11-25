# import numpy as np
# import pandas as pd

# # create data
# x = np.linspace(-10, 10, 100)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def d_sigmoid(x):
#     return sigmoid(x) * (1 - sigmoid(x))

# # get sigmoid output
# y = sigmoid(x)

# # get derivative of sigmoid
# d = d_sigmoid(x)

# df = pd.DataFrame({"x": x, "sigmoid(x)": y, "d_sigmoid(x)": d})

# print(df)

# df.to_csv('sigmoid', encoding='utf-8', index=False)

import numpy as np
import matplotlib.pyplot as plt
import random

x = np.array(range(0, 10))
y = np.array([100, 23, 44, 12, 55, 85, 69, 33, 75, 2])
z = y * random.randint(1,10)
plt.title("Plotting 1-D array")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color="red", marker="o", label="Array elements")
plt.legend()

plt.title("Plotting 1-D array")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, z, color="blue", marker="o", label="Array elements")
plt.legend()
plt.show()