# import numpy as np
#
# # Assuming your 2D array is named 'image_array'
# image_array = np.array([[0, 1, 0, 0, 0],
#                        [0, 0, 1, 0, 0],
#                        [0, 0, 1, 0, 0],
#                        [0, 0, 0, 1, 0],
#                        [0, 1, 0, 0, 0]])
#
# # Find indices where the value is greater than 0
# # non_zero_indices = np.where(image_array > 0)
# #
# # # Randomly select an index from the list of non-zero indices
# # random_index = np.random.choice(len(non_zero_indices[0]))
# #
# # # Get the corresponding pixel value using the randomly selected index
# # random_pixel_value = image_array[non_zero_indices[0][random_index], non_zero_indices[1][random_index]]
# #
# # print("Random Pixel Value:", random_pixel_value)
# # print("Pixel Coordinates:", (non_zero_indices[0][random_index], non_zero_indices[1][random_index]))
#
#
# np.where(image_array > 0)
#
# list_pixel = list(zip(*np.where(image_array > 0)))
#
# index = np.random.randint(0, len(list_pixel))
#
# choosen_pixel = list_pixel[index]
# print(choosen_pixel)
#
# x_pixel = choosen_pixel[0]
# y_pixel = choosen_pixel[1]


import matplotlib.pyplot as plt
import numpy as np

# Assuming X is your dataset containing images and error_label is the corresponding labels
# Replace these with your actual data
X = np.random.rand(10, 28 * 28)
error_label = np.random.randint(0, 10, 100)

fig, ax = plt.subplots()

# Display the first image with gray color scale
img_plot_gray = ax.imshow(255 - X[0].reshape(28, 28), cmap='gray')

# Display the second image with blue color scale
img_plot_blue = ax.imshow(X[0].reshape(28, 28), cmap='Blues', alpha=0.5)  # alpha for transparency

# annotation_string = ('Label = ' + str(error_label[0]) + '\n'
#                      + 'Output = ' + str(error_output_nn[0]) + '\n'
#                      + 'Error: ' + str(1) + ' of ' + str(len(images)))
#
# annotation = ax.annotate(annotation_string, xy=(0.79, 0.13), xycoords='figure fraction',
#                          horizontalalignment='right')

current_index = 0

def on_arrow_key(event):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(X)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(X)

    # Update both images
    img_plot_gray.set_data(255 - X[current_index].reshape(28, 28))
    img_plot_blue.set_data(X[current_index].reshape(28, 28))

    # You can uncomment this part if you have defined error_output_nn and images
    # annotation.set_text('Label = ' + str(error_label[current_index]) + '\n'
    #                     + 'Output = ' + str(error_output_nn[current_index]) + '\n'
    #                     + 'Error: ' + str(current_index + 1) + ' of ' + str(len(images)))

    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_arrow_key)
# plt.title('Accuracy: ' + str(percentage) + '%')
plt.show()


