from src.app.forward.forward import forward_training
from src.app.backpropagation.backpropagation import backpropagation_training

from src.app.validation.validation import forward_validation_single
from src.app.validation.validation_plot import forward_validation_graph

# Parameters
l = 60000  # Number of examples
ETA = 0.01  # learning rate
epochs = 2000  # epochs

# Learning method
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

# Training
forward_training(l, ETA, epochs, learning_mode)
# backpropagation_training(l, ETA, epochs, learning_mode)

# Validation
validation_dataset = 'mnist_test'
validation_threshold = 0.0

weight_and_biases_path = 'F-mini-l=60000-eta=0.01-epoch=1000/epoch=900'
# forward_validation_single(validation_dataset, weight_and_biases_path, validation_threshold)
# forward_validation_graph(validation_dataset, weight_and_biases_path, validation_threshold)

# weight_and_biases_path = 'B-mini-l=60000-eta=0.001-epoch=500/epoch=100'
# backprop_validation(validation_dataset, weight_and_biases_path, validation_threshold)
