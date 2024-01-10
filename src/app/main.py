from app.forward.forward import forward_training

from app.validation import forward_validation
from app.validation import backprop_validation


### Parameters
l = 60000  # Number of examples
ETA = 0.001  # learning rate
epochs = 1500  # epochs

# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

### Forward Training
forward_training(l, ETA, epochs, learning_mode)

# validation_dataset = 'mnist_test.csv'

### Forward Validation
# weight_and_biases_path = 'W-F-mini-l=60000-epoch=1000-eta=0.001/W-F-mini-l=60000-epoch=1000-eta=0.001'
# forward_validation(validation_dataset, weight_and_biases_path)

### Backpropagation Validation
# weight_and_biases_path = 'W-BP-mini-l=60-epoch=10'
# backprop_validation(validation_dataset, weight_and_biases_path)
