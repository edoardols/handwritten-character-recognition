
from forward.forward import forward_training

# PARAMETERS
l = 10  # Number of examples
ETA = 0.01  # learning rate
epochs = 2000  # epochs
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'


# FORWARD TRAINING
print('Forward: Start')

forward_training(l, ETA, epochs, learning_mode)

print('Forward: Done')

