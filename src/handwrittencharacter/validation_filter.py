# region IMPORTS
import os

from src.handwrittencharacter.validation.forward import forward_validation_single

from src.handwrittencharacter.validation.backprop import backpropagation_validation_single

from src.handwrittencharacter.validation.validation_confront import validation_confront_graph
from src.handwrittencharacter.validation.validation_confront import validation_confront_noise_graph
# endregion

PATH_MAIN_FILE = os.path.dirname(__file__)

# region SETTINGS

# What u want to do? ( validation / confrontation / confrontation_noise )
program_type = 'confrontation_noise'

# region Trained NN
# Type ( B / F )
Type = 'B'

# Learning method ( batch / mini / online )
learning_mode = 'mini'
batch_dimension = 128

# Parameters
l = 60000  # Number of examples
ETA = 0.001  # learning rate
epochs = 500  # epochs

STEP = 10  # for validation graph

# region Creating the path
if learning_mode == 'mini':
    weight_and_biases = Type + '-' + learning_mode + '=' + str(batch_dimension) + '-l=' + str(l) + '-eta=' + str(ETA)
else:
    weight_and_biases = Type + '-' + learning_mode + '-l=' + str(l) + '-eta=' + str(ETA)

weight_and_biases_path = weight_and_biases + '/epoch=' + str(epochs)

print('Object of the program: ' + weight_and_biases_path)
# endregion
# endregion

# region Validation library
classic = 'mnist_test'
# blob = 'blob/mnist_test-bl-p-0.2'
# brightness = 'brightness/mnist_test-br-p-0.3'
# obscure = 'obscure/mnist_test-ob'
# salt_pepper = 'salt_pepper/mnist_test-sp-s-0.5-p-0.6'
# thickness = 'thickness/mnist_test-th-step=2'

blob = 'blob/mnist_test-bl-p-0.2'
blob1 = 'blob/mnist_test-bl-p-0.3'
blob2 = 'blob/mnist_test-bl-p-0.4'
blob3 = 'blob/mnist_test-bl-p-0.5'
blob4 = 'blob/mnist_test-bl-p-0.8'

validation_threshold = 0.0

validation_array = [classic, blob, blob1, blob2, blob3, blob4]
# endregion

# region Confrontation library
# nn1 = 'B-batch-l=60000-eta=0.0001/epochs=' + str(epochs)
# nn2 = 'B-mini=128-l=60000-eta=0.1/epochs=' + str(epochs)
# nn3 = 'B-mini=1024-l=60000-eta=0.001/epochs=' + str(epochs)

nn1 = 'F-batch-l=60000-eta=0.01/epochs=' + str(epochs)
nn2 = 'F-batch-l=60000-eta=0.001/epochs=' + str(epochs)
nn3 = 'F-batch-l=60000-eta=0.0001/epochs=' + str(epochs)
nn4 = 'F-batch-l=60000-eta=0.00001/epochs=' + str(epochs)

nn_array = [nn1, nn2, nn3, nn4]
# endregion

# endregion

# region PROGRAM
if program_type == 'validation':
    if Type == 'F':
        for i in range(0, len(validation_array)):
            forward_validation_single(PATH_MAIN_FILE, validation_array[i], weight_and_biases_path, validation_threshold)
    elif Type == 'B':
        for i in range(0, len(validation_array)):
            backpropagation_validation_single(PATH_MAIN_FILE, validation_array[i], weight_and_biases_path,
                                              validation_threshold)
    else:
        print('Invalid Type: you write ' + Type + '\nTry: B --> Backpropagation or F --> Forward')
        exit()
elif program_type == 'confrontation':
    validation_confront_graph(PATH_MAIN_FILE, classic, nn_array, epochs, STEP, validation_threshold)
elif program_type == 'confrontation_noise':
    validation_confront_noise_graph(PATH_MAIN_FILE, validation_array, weight_and_biases_path, epochs, STEP, validation_threshold)
else:
    print('Invalid Type: you write ' + program_type + '\nTry: validation or confrontation')
    exit()

# endregion
