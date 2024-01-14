weight_and_biases_path = 'F-mini-l=60000-eta=0.01-epoch=1000/epoch=900'

folder = weight_and_biases_path.split('/')
print(folder)

parameters = folder[0].split('-')
print(parameters)

l = parameters[2].split('=')[1]
ETA = parameters[3].split('=')[1]
learning_mode = parameters[1]
epoch = folder[1].split('=')[1]

print('l:', l)
print('ETA:', ETA)
print('mode:', learning_mode)
print('epoch:', epoch)