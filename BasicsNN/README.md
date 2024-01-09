## Basics of Neutral Networks
### :question: What are they?
Neutral network  is a function detecting some pattern in input data. It consists of 3 main parts: input layer (data), hidden layers (magic box responsible for proceesing) and output layer (results). Each layer is built by neurons, that can be also treated as functions. So, neuron takes outputs of all the neurons in previous layer, calculates them by proper, dedicated weights and bias and as the result, spits out a number between 0 and 1 = neuron's activation. In this process, `activation function` plays a crucial role, because they are used to introduce nonlinearity to models, which allows deep learning models to learn nonlinear prediction boundaries. There are many types of activation functions: binary step function, linear and non-linear functions (tahn, sigmoid or relu = the most popular). Output layer has got also activation function - in case of binary prediction sigmoid is chosen, but for multi-class prediciton softmax is the best option.

In summary the NN is a function that involves parameters in form of weights and biases. Each layer tries to extract specific features. For examples, when it comes to images: the first few layers focuses on a details like edges, corners and curvers, but the last few layers detects more general features - the domimant objects. From particular to general.

### :mortar_board: How do they learn? 
In a nutshell, a NN learning is a process of finding the right weights and biases. It is done by `cost function`, telling the model wheather it perfoms correctly or not. The cost of a single training example (`loss function`) is calculated by quantifying the difference between predicted and actual outputs. Cost function is understood as the average of all loss function values. There are many popular functions depending on the problem: regression (MSE, MAE, Huber etc) and classification (binary cross-entropy, categorial cross-entropy etc). The goal is to minimalize the cost function, by changing weights of each hidden layer. It is done by calculating `gradient decent`, and `back propagation algorithm` is the most efficient method for computing it.

`Gradient decent algorithm` helps to figure out what is the downhill direction. This direction is the negative of the gradient of the cost function, and its lenght is a indicator of the slope's steepness.  Moreover, the magnitude of each component of calculated gradient vector tells how sensitive the cost function is to each weight and bias. The training process consists of calculating gradient, taking a small step downhill and just repeating that over and over. In this process, there are three main meaningful hyperparameters: `learning rate` controls the rate or speed at which the model learns = the amount of apportioned error that the weights of the model are updated, `momentum` helps to prevent oscillations, indicates how many gradients from the past are considered (higher = more), and `decay` adjusts learning rate (lr) per iterations according to equation, simpilfied one `lr = lr * (1. / (1. + decay * iterations))`. Definitely, it is a way to converge towards some local minimum of a cost function. Unfortunatelly,  our loss function usually has various local minima, which can missguide the model. In order to prevent it, we can manually monitor and fix learning rate parameter - but it is impossible. That is why we should set the `optimizer` parameter that does this for us. It optimizes the learning rate automatically to avoid entering a local minimum and is also responsible for fastening the optimization process. The most popular optimizers are Adam, RMS prop, Adagrad. 

There are also other important hyperparameters of the model training process, affecting both the accuracy and computational efficiency of the training process.
- `number of hidden layers` - one of the indicators of model complexity.
- `number of epochs` - determines how many times the model will see the entire training data before completing training.
- `batch_size` - represents the number of samples used in one forward and backward pass through the network. It can be understood as a trade-off between accuracy and speed.

After training model, you show it more labeled data (testing data), that it has never seen before. Then you can see how accurately the model classifies those images.

### :chart_with_upwards_trend: Metrics
There are several metrics helping in performance model assesment. Depending on task type (classification, segmentation), you should conclude based on proper indicator(s).

### :page_with_curl: Common issues
- Imbalanced data
In ideal scenario the data are balanced. Unfortunatelly, there is usually a problem with it. You want to get as balanced data as it is possible, but luckily there are a lot of popular techniques to hadle imbalanced data like resampling, loss functions with weights, cross-validation or using right evaluation metrics.

- Overfitting 

- Underfitting

 

Created by Agnieszka Florkowska, 2024
