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

inner = []
ax = []
dataset = []
X = []
plot = []

# region DATASET
dataset.append(pd.read_csv('../dataset/' + file_name + '.csv', header=None))
dataset.append(pd.read_csv('../dataset/blob/' + file_name + '-bl-p-' + str(p) + '.csv', header=None))
dataset.append(pd.read_csv('../dataset/brightness/' + file_name + '-br-p-' + str(p) + '.csv', header=None))
dataset.append(pd.read_csv('../dataset/obscure/' + file_name + '-ob' + '.csv', header=None))
dataset.append(pd.read_csv('../dataset/salt_pepper/' + file_name + '-sp-s-' + str(s) + '-p-' + str(p) + '.csv', header=None))
dataset.append(pd.read_csv('../dataset/thickness/' + file_name + '-th-step=' + str(step) + '.csv', header=None))
# endregion

fig = plt.figure()
outer = gridspec.GridSpec(2, 3)
for i in range(0, 6):
    inner.append(gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[i]))

    ax.append(plt.Subplot(fig, inner[i][0]))

    X_D = dataset[i].iloc[:, 1:]
    X.append(X_D.to_numpy())

    plot.append(ax[i].imshow(X[i][0].reshape(28, 28), cmap='gray', vmin=0, vmax=255))  # for the original version
    fig.add_subplot(ax[i])

    current_index = 0

def on_arrow_key(event, ):
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(X[0])
    elif event.key == 'left':
        current_index = (current_index - 1) % len(X[0])

    # Update images
    for i in range(0, 6):
        plot[i].set_data(X[i][current_index].reshape(28, 28))

    fig.canvas.draw()


fig.canvas.mpl_connect('key_press_event', on_arrow_key)

# region TITLES
ax[0].set_title('Classic')
ax[1].set_title('Blob')
ax[2].set_title('Brightness')
ax[3].set_title('Obscure')
ax[4].set_title('Salt & Pepper')
ax[5].set_title('Thickness')
# endregion

plt.show()
