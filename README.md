# Getting started

This repo contains configuration instruction as well as the explanation of main terms associated with neural networks. README file decribes the theory and the proper .py files concern the pratical part.

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

# Neutral networks

### What are they?
Neutral network  is a function detecting some pattern in input data. It consists of 3 main parts: input layer (data), hidden layers (magic box responsible for proceesing) and output layer (results). Each layer is built by neurons, that can be also treated as functions. So, neuron takes outputs of all the neurons in previous layer, calculates them by proper, dedicated weights and bias and as the result, spits out a number between 0 and 1 = neuron's activation. In this process, `activation function` plays a crucial role, because they are used to introduce nonlinearity to models, which allows deep learning models to learn nonlinear prediction boundaries. There are many types of activation functions: binary step function, linear and non-linear functions (tahn, sigmoid or relu = the most popular). Output layer has got also activation function - in case of binary prediction sigmoid is chosen, but for multi-class prediciton softmax is the best option.

In summary the NN is a function that involves parameters in form of weights and biases. Each layer tries to extract specific features. For examples, when it comes to images: the first few layers focuses on a details like edges, corners and curvers, but the last few layers detects more general features - the domimant objects. From particular to general.

### How do they learn? 
In a nutshell, a NN learning is a process of finding the right weights and biases. It is done by cost function, telling the model wheather it perfoms correctly or not. The cost of a single training example is calculated by the square of difference of network's outputs and the expected values. The goal is to minimalize the cost function, by changing weights of each hidden layer. It is done by calculating `gradient decent`, and `Back propagation algorithm` is the most efficient method for computing it.

`Gradient decent algorithm` helps to figure out what is the downhill direction. This direction is the negative of the gradient of the cost function, and its lenght is a indicator of the slope's steepness. So the training process consists of calculating gradient, taking a small step downhill (its is determined by `learning rate` hyperparameter) and just repeating that over and over. It is a way to converge towards some local minimum of a cost function. Another important thing to know is that the magnitude of each component of calculated gradient vector tells how sensitive the cost function is to each weight and bias. Unfortunatelly,  our loss function usually has various local minima, which can missguide the model. In order to prevent it, we can manually monitor and fix learnig rate parameter - but it is impossible. That is why we should set the `optimizer` paramater that does this for us. It optimizes the learning rate automatically to avoid entering a local minimum and is also responsible for fastening the optimization process. The most popular optimizers are Adam, RMS prop, Adagrad. 

After training model, you show it more labeled data (testing data), that it has never seen before. Then you can see how accurately the model classifies those images.

There are also other important hyperparameters of the model training process, affecting both the accuracy and computational efficiency of the training process.
- `number of hidden layers` - one of the indicators of model complexity 
- `momentum` - helps to prevent oscillations, indicate how many gradients from the past (history) are considered (higher = more).
- `number of epochs` - determines how many times the model will see the entire training data before completing training.
- `batch_size` - represents the number of samples used in one forward and backward pass through the network. It can be understood as a trade-off between accuracy and speed. 

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

