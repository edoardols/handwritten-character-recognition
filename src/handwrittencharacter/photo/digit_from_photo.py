import os

import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.handwrittencharacter.photo.validation_photo import forward_photo_validation, backpropagation_photo_validation


def select_file():
    root = tk.Tk()

    # Make the Tk window visible
    root.attributes('-topmost', True)
    root.attributes('-fullscreen', True)
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(title="Select a photo",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
    return file_path


def get_user_text():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    user_text = simpledialog.askstring("Input", "What number it is?")
    root.destroy()  # Destroy the window after getting the input
    return user_text


PATH_MAIN_FILE = os.path.dirname(__file__)

# Load the image
image = Image.open(select_file())

label = get_user_text()

# Convert to black and white (grayscale)
image_bw = image.convert("L")

# Invert pixel values (black becomes white and white becomes black)
image_bw_inverted = Image.eval(image_bw, lambda x: 255 - x)

# Resize to 28x28 pixels
image_resized = image_bw_inverted.resize((28, 28))

# Convert to numpy array
digit = (np.array(image_resized)).reshape(1, -1)

weight_and_biases_path = 'F-mini=512-l=60000-eta=0.0001/epoch=2500/'
forward_output = forward_photo_validation(PATH_MAIN_FILE, weight_and_biases_path, digit)

weight_and_biases_path = 'B-mini=1024-l=60000-eta=0.01/epoch=500/'
backpropagation_output = backpropagation_photo_validation(PATH_MAIN_FILE, weight_and_biases_path, digit)

# Create a grid layout for the main figure
gs = GridSpec(2, 3, height_ratios=[4, 4], hspace=0.3)

# Plotting
fig = plt.figure(figsize=(12, 6))

# Original Image
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

# Resized Image
ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(image_resized, cmap='gray')
ax2.set_title('Resized Image (28x28)')
ax2.axis('off')

# Mr Supervisor
ax3 = fig.add_subplot(gs[1, 0])
image = Image.open(PATH_MAIN_FILE + '/mr-supervisor.png')
ax3.imshow(image)
ax3.set_title("Mr Supervisor says: \"It's a " + label + '"')
ax3.axis('off')

# Forward Neural Network
ax4 = fig.add_subplot(gs[1, 1])
image = Image.open(PATH_MAIN_FILE + '/FeedForward_NN.png')
ax4.imshow(image)
ax4.set_title("Forward NN says: \"It's a " + str(forward_output) + '"')
ax4.axis('off')

# Backpropagation Neural Network
ax5 = fig.add_subplot(gs[1, 2])
image = Image.open(PATH_MAIN_FILE + '/BackPropagation_NN.png')
ax5.imshow(image)
ax5.set_title("Backprop NN says: \"It's a " + str(backpropagation_output) + '"')
ax5.axis('off')

plt.show()
