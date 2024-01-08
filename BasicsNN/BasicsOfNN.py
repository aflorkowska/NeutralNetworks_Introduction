# Ignoring printing warnings and info. Still showing errors.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import plotMethods

# Using GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Data loading and statistics
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Training data size: {x_train.shape[0]}")
print(f"Testing data size: {x_test.shape[0]}")
print(f"Single input data dimension (= image size): {x_test.shape[1]} x {x_test.shape[2]}")
num_classes = np.size(np.unique(y_train))
print(f"Number of classes {num_classes}")

trainDistribution = plotMethods.bar(np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1], "Distribution of the numbers training data of various classes")
trainDistribution.show()
testDistribution = plotMethods.bar(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1], "Distribution of the numbers testing data of various classes")
testDistribution.show()

trainSetSamples = plotMethods.samplesOfSet(x_train, y_train, "Training set - examples", 2, 5, 0, 10)
trainSetSamples.show()
testSetSamples = plotMethods.samplesOfSet(x_test, y_test, "Testing set - examples", 2, 5, 0, 10)
testSetSamples.show()

# Preprocessing

# Model

# Training

# Evaluation
