from src.app.forward.forward import forward_training
from src.app.backpropagation.backpropagation import backpropagation_training

from src.app.validation.validation import forward_validation
from src.app.validation.validation_plot import forward_validation as for_val

# region Settings

# Parameters
l = 60000  # Number of examples
ETA = 0.01  # learning rate
epochs = 1000  # epochs

# Learning method
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

# endregion

# region Forward

# Training
forward_training(l, ETA, epochs, learning_mode)

# Validation
# weight_and_biases_path = 'W-F-mini-l=60000-epoch=500-eta=0.001/W-F-mini-l=60000-epoch=500-eta=0.001'
# weight_and_biases_path = 'W-F-online-l=60000-epoch=500-eta=0.001/W-F-online-l=60000-epoch=400-eta=0.001'
# forward_validation(validation_dataset, weight_and_biases_path, validation_threshold)

# Validation dataset
# validation_dataset = 'mnist_test'

# # Parameters
# l = 60000  # Number of examples
# ETA = 0.001  # learning rate
# epochs = 2500  # epochs
#
# validation_threshold = 0.2
#
# # Learning method
# # learning_mode = 'batch'
# learning_mode = 'mini'
# # learning_mode = 'online'
#
# for_val(validation_dataset, learning_mode, l, epochs, ETA, validation_threshold)
# # endregion
#
# # region Backpropagation
#
# # Training
# # backpropagation_training(l, ETA, epochs, learning_mode)
#
# # Validation
# # weight_and_biases_path = 'W-B-mini-l=60000-epoch=500-eta=0.001/W-B-mini-l=60000-epoch=100-eta=0.001'
# # backprop_validation(validation_dataset, weight_and_biases_path, validation_threshold)
#
# # endregion

