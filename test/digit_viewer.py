import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset from CSV
mnist_data = pd.read_csv(
    'C:\\Users\\ELasc\\PycharmProjects\\handwritten-character-recognition\\dataset\\mnist_test.csv', header=None)

# Extract images and labels
images = mnist_data.iloc[:, 1:]
labels = mnist_data.iloc[:, :1]

global current_index
current_index = 0

fig, ax = plt.subplots()

# Display the first image
img_plot = plt.imshow(images[0].reshape(28, 28), cmap='gray')


def on_arrow_key(event, ):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(images)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(images)

    img_plot.set_data(1 - images[current_index].reshape(28, 28))

    annotation.set_text('Error: ' + str(current_index + 1) + ' of ' + str(len(images)))
    fig.canvas.draw()


#fig.canvas.mpl_connect('key_press_event', on_arrow_key)
plt.title('Accuracy: 0%')
plt.show()
