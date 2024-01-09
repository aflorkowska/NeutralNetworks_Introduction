# Ignoring printing warnings and info. Still showing errors.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Sequential vs Functional API
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Using GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

