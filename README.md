<img width="592" alt="image" src="https://github.com/edoardols/handwritten-character-recognition/assets/150338322/ebaef4d4-44e0-45b8-95ee-098a01c767cc">
A Machine Learning project for the ML course at the University of Siena
Goal of the project: Train MNIST dataset with two types of neural networks (in our specific case, we trained two types of Neural Networks: one with input-output layers and the other with two hidden layers with 16 neurons each)
1)In the "dataset" folder you will find the MNIST dataset loaded (training set and validation set) and the noised dataset (Blob, Brightness, Obscure, Salt and pepper, Thickness) where the way how noise is added can be found in the noise folder contained in handwrittencharacter folder.
2) In the "handwrittencharacter" folder you can find the training folders (backpropagation and forward) where also trained data are present; while to run training/validation selecting a trained dataset, can be done in main initializing all parameters. To produce graph comparison, validation_filter contains all plot functions.
#############################################################################################################
src/handwrittencharacter/backpropagation/training -> trained datasets of backpropagation Neural network
src/handwrittencharacter/forward/training -> trained datasets of forward Neural network
src/handwrittencharacter/main.py -> the main to make training and validation (single graph)
src/handwrittencharacter/validation_filter.py -> graph comparison
src/XOR -> to see XOR test on Backpropagation Neural Network

#############################################################################################################
# handwritten-character-recognition
A Machine Learning project for the ML course of the University of Siena

## Installation guide for TensorFlow for windows

[TensorFlow official guide](https://www.tensorflow.org/install/pip?hl=it)

### Required for TensorFlow (CPU)
[Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
pip >= 19.0
[Visual Studio](https://visualstudio.microsoft.com/it/)
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/it-IT/cpp/windows/latest-supported-vc-redist?view=msvc-170)

### Required for TensorFlow (Nvidia GPU)
Nvidia Driver
[CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
[SDK cuDNN 8.6.0 for CUDA 11.x](https://developer.nvidia.com/rdp/cudnn-archive)

Install cuDNN

C: OS > Program Files > Nvidia GPU Computing Toolkit > CUDA > v11.8 > copy

Start > "env" > Edit the system environment variables > add folder /bin and /libnvvp to "Path"

Youtube video
[1](https://www.youtube.com/watch?v=IubEtS2JAiY)
[2](https://www.youtube.com/watch?v=ctQi9mU7t9o)
