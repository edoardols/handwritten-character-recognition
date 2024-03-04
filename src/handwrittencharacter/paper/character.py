import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = 'C:\\Users\\ELasc\\PycharmProjects\\handwritten-character-recognition\\dataset\\'

dataset = 'mnist_test.csv'
# dataset = 'blob/mnist_test-bl-p-0.2.csv'
# dataset = 'brightness/mnist_test-br-p=0.5.csv'
# dataset = 'obscure/mnist_test-ob.csv'
# dataset = 'salt_pepper/mnist_test-sp-s-1-p-0.3.csv'
# dataset = 'thickness/mnist_test-th-step=1.csv'

validation_dataset = pd.read_csv(path + dataset, header=None)

XV_D = validation_dataset.iloc[:, 1:]
X = XV_D.to_numpy()

YV_D = validation_dataset.iloc[:, :1]
YV = YV_D[0].to_numpy()

fig, ax = plt.subplots()

# Display the first image
img_plot = ax.imshow(X[35].reshape(28, 28), cmap='gray')


# annotation_string = ('Label = ' + str(error_label[0]) + '\n'
#                      + 'Output = ' + str(error_output_nn[0]) + '\n'
#                      + 'Error: ' + str(1) + ' of ' + str(len(images)))
#
# annotation = ax.annotate(annotation_string, xy=(0.79, 0.13), xycoords='figure fraction',
#                          horizontalalignment='right')

current_index = 35


def on_arrow_key(event, ):
    global current_index
    if event.key == 'right':
        current_index = current_index + 1
    elif event.key == 'left':
        current_index = current_index - 1

    img_plot.set_data(X[current_index].reshape(28, 28))
    fig.canvas.draw()


fig.canvas.mpl_connect('key_press_event', on_arrow_key)
# plt.axis('off')
plt.show()

# Select 25 random images
random_indices = np.array(range(0,4*8))
images = X[random_indices]

# Plot images in a 5x5 grid
fig, axes = plt.subplots(4, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].reshape(28,28), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()