# Info

This repo contains configuration instructions as well as the explanation of main terms associated with neural networks. README file decribes the theory and the proper .py files concern the pratical part.

# Configuration - connection with GPU

There are some crucial steps you should follow to properly install tensorflow and use CUDA toolkit.

1. Check compute capability for your GPU - [website](https://developer.nvidia.com/cuda-gpus).
2. Download and install specific driver for your GPU - [website](https://www.nvidia.com/Download/index.aspx).
3. Download and install Anaconda - [website](https://www.anaconda.com/download) - Do not forget to run .exe file as administrator.
4. Run as administrator (everytime!) `Anaconda Promt` window.
5. Create new enviroment for deep learning using command `conda create --name tf tensorflow-gpu`
6. Download and install PyCharm, Community Edition  - [website](https://www.jetbrains.com/pycharm/download/) - Do not forget to run .exe file as administrator.
7. Set proper Python interpreter.
```
Settings -> Python Interpreter -> Add Interpreter -> Add Local Interpreter-> Conda Enviroment -> Set path for the conda.exe file (in Scripts directory) -> Load environments-> Set suitable interpreter from the list (dedicated for tf env)
```
8. Check if tensorflow-gpu was installed successfully
```
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
```

# Recommended materials 
[Mathematical explanation of neutral networks, what they are, how they learn by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

# Neutral networks - main idea 

### What are they?
Network is a function detecting some pattern in input data. It consists of 3 main parts: input layer (data), hidden layers (magic box responsible for proceesing) and output layer (results). Each layer is built by neurons, that can be also treated as functions. So, neuron takes outputs of all the neurons in previous layer, calculates them by proper, dedicated weights and bias and as the result, spits out a number between 0 and 1 = neuron's activation. In this process, neuron's activation function plays a crucial role. There are many types of activation functions: binary step function, linear and non-linear functions (tahn, sigmoid or relu = the most popular). 

In summary the NN is a function that involves parameters in form of weights and biases. Each layern tries to extract specific features. For examples, when it comes to images: the first few layers focuses on a details like edges, corners and curvers, but the last few layers detects more general features - the domimant objects. From particular to general.

### How do they learn? 
In a nutshell, a NN learning is a process of finding the right weights and biases. It is done by cost function, telling the model wheather it perfoms correctly or not. The cost of a single training example is calculated by the square of difference of network's outputs and the expected values. The goal is to minimalize the cost function, by changing weights of each hidden layer. It is done by calculating `Gradient Decent`, and `Back propagation algorithm` is the most efficient method for computing it.

`Gradient decent algorithm` helps to figure out what is the downhill direction. This direction is the negative of the gradient of the cost function, and its lenght is a indicator of the slope's steepness. So the training process consists of calculating gradient, taking a small step downhill (its is determined by `learning rate` hyperparameter) and just repeating that over and over. It is a way to converge towards some local minimum of a cost function. Another important thing to know is that the magnitude of each component of calculated gradient vector tells how sensitive the cost function is to each weight and bias. Unfortunatelly,  our loss function usually has various local minima, which can missguide the model. In order to prevent it, we can manually monitor and fix learnig rate parameter - but it is impossible. That is why we should set the `optimizer` paramater that does this for us. It optimizes the learning rate automatically to avoid entering a local minimum and is also responsible for fastening the optimization process. The most popular optimizers are Adam, RMS prop, Adagrad. 

After training model, you show it more labeled data (testing data), that it has never seen before. Then you can see how accurately the model classifies those images.

# Common issues
### Imbalanced data
In ideal scenario the data are balanced. Unfortunatelly, there is usually a problem with it. You want to get as balanced data as it is possible, but luckily there are a lot of popular techniques to hadle imbalanced data like resampling, loss functions with weights, cross-validation or using right evaluation metrics.

### Overfitting 

### Underfitting

# Metrics
There are several metrics helping in performance model assesment. Depending on task type (classification, segmentation), you should conclude based on proper indicator(s).

# Convolutional Neutral Networks

# Recurrent Neutral Networks

# Transfer Learning 

# Fine tuning 

# Vision Transformers

# Vision Graph Neutral Networks

