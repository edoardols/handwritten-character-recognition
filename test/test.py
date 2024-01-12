#region Settings

# Parameters
l = 60000  # Number of examples
ETA = 0.001  # learning rate
epochs = 200  # epochs

# Learning method
# learning_mode = 'batch'
learning_mode = 'mini'
# learning_mode = 'online'

# Validation dataset
validation_dataset = 'XOR_val.csv'

#endregion

#region Forward
#
# # Training
# forward_training(l, ETA, epochs, learning_mode)
#
# # Validation
# # weight_and_biases_path = 'W-F-mini-l=60000-epoch=500-eta=0.001/W-F-mini-l=60000-epoch=500-eta=0.001'
# weight_and_biases_path ='W-F-online-l=60000-epoch=500-eta=0.001/W-F-online-l=60000-epoch=500-eta=0.001'
# forward_validation(validation_dataset, weight_and_biases_path)

#endregion

#region Backpropagation

# Training
#backpropagation_training(l, ETA, epochs, learning_mode)

# Validation
weight_and_biases_path = 'W-B-mini-l=60000-epoch=500-eta=0.001/W-B-mini-l=60000-epoch=100-eta=0.001'
#backprop_validation(validation_dataset, weight_and_biases_path)

#endregion