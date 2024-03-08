# region IMPORTS
import os

from src.handwrittencharacter.forward.forward import forward_training
from src.handwrittencharacter.backpropagation.backpropagation import backpropagation_training

from src.handwrittencharacter.validation.forward import forward_validation_single
from src.handwrittencharacter.validation.forward import forward_validation_graph

from src.handwrittencharacter.validation.backprop import backpropagation_validation_single
from src.handwrittencharacter.validation.backprop import backpropagation_validation_graph
# endregion

PATH_MAIN_FILE = os.path.dirname(__file__)

# region SETTINGS
# Type ( B / F )
Type = 'B'

# Program ( train / validation / graph / all )
program = 'all'

# Parameters
l = 60000  # Number of examples
ETA = 0.0001  # learning rate
epochs = 1000  # epochs
STEP = 10  # for validation graph

# Learning method ( batch / mini / online )
learning_mode = 'batch'
batch_dimension = 1024

# Validation ( mnist_test )
validation_dataset = 'mnist_test'
validation_threshold = 0.0
# endregion

# region Program
# NN that u want to validate
if learning_mode == 'mini':
    weight_and_biases = Type + '-' + learning_mode + '=' + str(batch_dimension) + '-l=' + str(l) + '-eta=' + str(ETA)
else:
    weight_and_biases = Type + '-' + learning_mode + '-l=' + str(l) + '-eta=' + str(ETA)

weight_and_biases_path = weight_and_biases + '/epoch=' + str(epochs)

print('Object of the program: ' + weight_and_biases_path)

if Type == 'F':
    if program == 'train':
        forward_training(PATH_MAIN_FILE, l, ETA, epochs, learning_mode, batch_dimension)
    elif program == 'validation':
        forward_validation_single(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, validation_threshold)
    elif program == 'graph':
        forward_validation_graph(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, validation_threshold)
    elif program == 'all':
        forward_training(PATH_MAIN_FILE, l, ETA, epochs, learning_mode, batch_dimension)
        forward_validation_single(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, validation_threshold)
        forward_validation_graph(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, validation_threshold)
    else:
        print('Wrong settings: try use one of the following --> train / validation / both')
        exit()

elif Type == 'B':
    if program == 'train':
        backpropagation_training(PATH_MAIN_FILE, l, ETA, epochs, learning_mode, batch_dimension)
    elif program == 'validation':
        backpropagation_validation_single(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path,
                                          validation_threshold)
    elif program == 'graph':
        backpropagation_validation_graph(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, epochs, STEP,
                                         validation_threshold)
    elif program == 'all':
        backpropagation_training(PATH_MAIN_FILE, l, ETA, epochs, learning_mode, batch_dimension)
        backpropagation_validation_single(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path,
                                          validation_threshold)
        backpropagation_validation_graph(PATH_MAIN_FILE, validation_dataset, weight_and_biases_path, epochs, STEP,
                                         validation_threshold)
    else:
        print('Wrong settings: try use one of the following --> train / validation / graph / all / filter')
        exit()
else:
    print('Wrong settings: try use one of the following --> F / B')
    exit()
# endregion
