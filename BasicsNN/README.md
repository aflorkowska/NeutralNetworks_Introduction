## Basics of Neutral Networks
### :question: What are they?

Neutral network (NN) is a function detecting some patterns (rules) in input data. In other words, it eliminates the need for a priori programming, enanbling selected process to become fully automated.

NN find their use in popular groups of tasks like:
- classification - assigning into defined target categories,
- detection - determining the position of the classified objects,
- segmentation - diving into groups with similar characteristics,

in many fields e.g. biomedical engineering, automotive industry (autonomous behaviors) or linguistics. 

Deep learning can be understood as class of multilayer structures (NNs), consisting of 3 main parts: input layer (data), hidden layers (magic box responsible for proceesing) and output layer (results). These layers are responsible for performing different transformations and operations, like adaptive filtration, selection, regularization and normalization, that will be introducted in next paragraphs.
  
<IMAGE>

### :mortar_board: How do they learn? 

First of all, there are two main way of NN learning: supervised that takes as input data with correspodning labels, and unsupervised, that on the contrary bases only on data to detect some common characteristic, aiming to cluster objects. Let's firstly dive into supervised method.

In this case, learning is treated as gaining ability to map correctly inputs to outputs, so it's a process of establishing rules.  

What does it mean indeed - "learning of mapping inputs to output"?

Before we start, it's crucial to remain that in computer's world it's all about numbers. Single word can be encoded as number, image can be understood as matrix of numbers. For example, RGB image refers to 3D matrix that defines red, green and blue components for each individual pixel. In case of grayscale image, matrix limits only to one component.   

<IMAGE>

`McCullocha-Pitts neuron` - one of the mathematical neuron's model. It's consist of many inputs, each of which is assigned a real number (`weight`), and one output. Moreover, added value is called as `bias` and used for offset the result. From mathematical point of view, single neuron is a function - linear transformation. It takes inputs, calculates them by weights, adds bias, computes the value of `activation function` of the determined sum, and as the result, spits out a number between 0 and 1. In this process, activation function plays a crucial role to ensure well model generalization and accurate prediction. It introduces nonlinearity to models, which allows deep learning models to learn highly complex patterns. There are many types of them: binary step function, linear and non-linear functions (tahn, sigmoid or ReLU). In case of binary classification, sigmoid method should be chosen as the activation function of output layer, but for multi-class classification softmax function. 

The McCulloch–Pitts neuron is the basic building block of a `Perceptron NN` (a simple supervised binary classifier). In turn, many perceptrons placed in one layer creates a "Fully Connected (Dense) Layer", in which all neurons of successive layers are connected to each other.

Okey, so that's the NN archicture, but still - how does this learning look like?

NN learning is an iterative process. Each iteration means doing some calculations by network, its performance evaluation and weights modification based on the result. In a nutshell, a NN learning is a process of finding the right parameters: weights and biases. 

But...how this NN knows, how to modify these weights to get better results?

It is done by `lost function`, telling the model wheather it perfoms correctly or not. The cost of a single training example is calculated by quantifying the difference between predicted and actual outputs. `Cost function` is understood as the average of all loss function values. There are many popular functions depending on the problem: regression (MSE, MAE, Huber) and classification (binary cross-entropy, categorial cross-entropy etc). The goal is to minimalize the cost function, by changing weights of each hidden layers. It is done by calculating `gradient decent`, and `back propagation algorithm` is the most efficient method for computing it.

`Gradient decent algorithm` helps to figure out what is the downhill direction. This direction is the negative of the gradient of the cost function, and its magnitude is a indicator of the slope's steepness = rate of change.  Moreover, the magnitude of each component of calculated gradient vector tells how sensitive the cost function is to each weight and bias. The training process consists of calculating gradient, taking a small step downhill (weights modification) and just repeating that over and over till finding minimum of the cost function (local or global). In this process, there are three main meaningful hyperparameters: `learning rate` meaning the amount of apportioned error that the weights of the model are updated, `momentum` helping to prevent oscillations, indicating how many gradients from the past are considered (higher = more), and `decay` adjusting learning rate (lr) per iterations. Setting proper values of these parameters makes it easier to converge towards some local minimum of a cost function. Unfortunatelly,  our loss function usually has various local minima, which can missguide the model. In order to prevent it, we can manually monitor and fix learning rate parameter - but it is impossible. That is why we should set the `optimizer` parameter that does this for us. It optimizes the learning rate automatically to avoid entering a local minimum and is also responsible for fastening the optimization process. The most popular optimizers are Adam, RMS prop, Adagrad. 


What is the best way to verify if model trained correctly?
After training model, there is a evaluation phrase. You show it more labeled data (testing data), that it has never seen before. Then you can see how accurately the model classifies those images.

The main clue is to correctly preprocess data and choose network architecture and its parameters = determine model complexity. It depends only on you, how many layers the network will have, what type of activation functions, loss functions, optimizers...and so on. 

________________________________

### :chart_with_upwards_trend: Metrics

There are several metrics helping in performance model assesment. Depending on task type, you should conclude based on proper indicator(s).
Accuracy, Recall, sensivity, ROC curves, Confusion matrix, dice (IoU), hausdorff coef. 

### :page_with_curl: Common issues

- Unrepresentative data

- Imbalanced data
In ideal scenario the data are balanced. Unfortunatelly, there is usually a problem with it. You want to get as balanced data as it is possible, but luckily there are a lot of popular techniques to hadle imbalanced data like resampling, loss functions with weights, cross-validation or using right evaluation metrics.

- Overfitting 

- Underfitting

 
## Practical part explanation

- `Batch size parameter` represents the number of samples used in one forward and backward pass through the network.
- `Number of epochs parameter` determines how many times the model will see the entire training data before completing training.

Created by Agnieszka Florkowska, 2024
