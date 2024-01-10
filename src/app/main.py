from app.validation import forward_validation
from app.validation import backprop_validation


### Parameters
l = 60  # Number of examples
ETA = 0.01  # learning rate
epochs = 50  # epochs

# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

### Forward Training
# forward_training(l, ETA, epochs, learning_mode)

validation_dataset = 'mnist_test.csv'

### Forward Validation
# weight_and_biases_path = 'W-F-mini-l=60-epoch=5000-eta=0.01/W-F-mini-l=60-epoch=1000-eta=0.01/'
# forward_validation(validation_dataset, weight_and_biases_path)

### Backpropagation Validation
weight_and_biases_path = 'W-BP-mini-l=60-epoch=10'
backprop_validation(validation_dataset, weight_and_biases_path)
