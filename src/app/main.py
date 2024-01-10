
from forward.validation import forward_validation
from forward.forward import forward_training

# Parameters
l = 60  # Number of examples
ETA = 0.01  # learning rate
epochs = 50  # epochs

# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

# Forward Training
# forward_training(l, ETA, epochs, learning_mode)

# Forward Validation

validation_dataset = 'mnist_test.csv'
weight_and_biases_path = 'W-F-mini-l=60-epoch=5000-eta=0.01/W-F-mini-l=60-epoch=1000-eta=0.01/'

forward_validation(validation_dataset, weight_and_biases_path)
