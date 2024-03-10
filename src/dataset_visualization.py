from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("TkAgg")  # Usa il backend TkAgg

# region SETTINGS

file_name = 'mnist_test'
p = 0.3
s = 0.5
step = 2

# endregion

# region FIGURE/SUBPLOTS
fig = plt.figure()
outer = gridspec.GridSpec(2, 3)

inner_classic = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
inner_blob = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
inner_brightness = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])
inner_obscure = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[3])
inner_salt_pepper = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[4])
inner_thickness = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[5])

ax_classic = plt.Subplot(fig, inner_classic[0])
ax_blob = plt.Subplot(fig, inner_blob[0])
ax_brightness = plt.Subplot(fig, inner_brightness[0])
ax_obscure = plt.Subplot(fig, inner_obscure[0])
ax_salt_pepper = plt.Subplot(fig, inner_salt_pepper[0])
ax_thickness = plt.Subplot(fig, inner_thickness[0])

dataset = pd.read_csv('../dataset/' + file_name + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_classic = X_D.to_numpy()

dataset = pd.read_csv('../dataset/blob/' + file_name + '-bl-p-' + str(p) + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_blob = X_D.to_numpy()

dataset = pd.read_csv('../dataset/brightness/' + file_name + '-br-p-' + str(p) + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_brightness = X_D.to_numpy()

dataset = pd.read_csv('../dataset/obscure/' + file_name + '-ob' + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_obscure = X_D.to_numpy()

dataset = pd.read_csv('../dataset/salt_pepper/' + file_name + '-sp-s-' + str(s) + '-p-' + str(p) + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_salt_pepper = X_D.to_numpy()

dataset = pd.read_csv('../dataset/thickness/' + file_name + '-th-step=' + str(step) + '.csv', header=None)
X_D = dataset.iloc[:, 1:]
X_thickness = X_D.to_numpy()

# Display the first image with gray color scale
plot_classic = ax_classic.imshow(X_classic[0].reshape(28, 28), cmap='gray')      # for the original version
fig.add_subplot(ax_classic)

plot_blob = ax_blob.imshow(X_blob[0].reshape(28, 28), cmap='gray')      # for the original version
fig.add_subplot(ax_blob)

plot_brightness = ax_brightness.imshow(X_brightness[0].reshape(28, 28), cmap='gray', vmin=0, vmax=255)      # for the original version
fig.add_subplot(ax_brightness)

plot_obscure = ax_obscure.imshow(X_obscure[0].reshape(28, 28), cmap='gray')      # for the original version
fig.add_subplot(ax_obscure)

plot_salt_pepper = ax_salt_pepper.imshow(X_salt_pepper[0].reshape(28, 28), cmap='gray')      # for the original version
fig.add_subplot(ax_salt_pepper)

plot_thickness = ax_thickness.imshow(X_thickness[0].reshape(28, 28), cmap='gray')      # for the original version
fig.add_subplot(ax_thickness)

current_index = 0

def on_arrow_key(event, ):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(X_classic)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(X_classic)

    # Update both images
    plot_classic.set_data(X_classic[current_index].reshape(28, 28))
    plot_blob.set_data(X_blob[current_index].reshape(28, 28))
    plot_brightness.set_data(X_brightness[current_index].reshape(28, 28))
    plot_obscure.set_data(X_obscure[current_index].reshape(28, 28))
    plot_salt_pepper.set_data(X_salt_pepper[current_index].reshape(28, 28))
    plot_thickness.set_data(X_thickness[current_index].reshape(28, 28))

    fig.canvas.draw()


fig.canvas.mpl_connect('key_press_event', on_arrow_key)

ax_classic.set_title('Classic')
ax_blob.set_title('Blob')
ax_brightness.set_title('Brightness')
ax_obscure.set_title('Obscure')
ax_salt_pepper.set_title('Salt & Pepper')
ax_thickness.set_title('Thickness')

plt.show()
# endregion