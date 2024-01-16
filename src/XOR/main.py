
from src.XOR.backpropagation.backpropagation import XOR_training

from src.XOR.validation.validation import XOR_validation_single
from src.XOR.validation.validation_plot import XOR_validation_graph

# Parameters
l = 60000  # Number of examples
ETA = 0.01  # learning rate
epochs = 100  # epochs

# Learning method
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

# Training
# XOR_training(l, ETA, epochs, learning_mode)

# Validation
validation_dataset = 'XOR_test'
validation_threshold = 0.0

weight_and_biases_path = 'B-mini-l=60000-eta=0.01-epoch=500/epoch=100'

XOR_validation_single(validation_dataset, weight_and_biases_path)
# XOR_validation_graph(validation_dataset, weight_and_biases_path)
