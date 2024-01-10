import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a list of 28x28 pixel images (represented as NumPy arrays)
# Replace this with your actual list of images
images = [np.random.rand(28, 28) for _ in range(5)]  # Example images

fig, ax = plt.subplots()
current_index = 0  # Initial index of the displayed image

# Display the first image
img_plot = ax.imshow(images[current_index], cmap='gray')

def on_arrow_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(images)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(images)
    img_plot.set_data(images[current_index])
    fig.canvas.draw()

# Connect the key press event to the function
fig.canvas.mpl_connect('key_press_event', on_arrow_key)

plt.show()

