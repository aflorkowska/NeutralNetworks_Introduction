# Ignoring printing warnings and info. Still showing errors.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Sequential vs Functional API
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# Sequential API (Very convenient, not very flexible)
modelA = keras.Sequential(
   [
       keras.Input(shape=(28 * 28)),
       layers.Dense(512, activation="relu"),
       layers.Dense(256, activation="relu"),
       layers.Dense(10),
   ]
)

modelB = keras.Sequential()
modelB.add(keras.Input(shape=(784)))
modelB.add(layers.Dense(512, activation="relu"))
modelB.add(layers.Dense(256, activation="relu", name="my_layer"))
modelB.add(layers.Dense(10))

# Functional API (A bit more flexible, allowing for more complex models with multiple inputs or outputs, shared layers, and non-linear flows)
inputs = keras.Input(shape=(784))
xA = layers.Dense(512, activation="relu", name="first_layer")(inputs)
xA = layers.Dense(256, activation="relu", name="second_layer")(xA)
xB = layers.Dense(128, activation="relu", name="third_layer")(xA)
outputsA = layers.Dense(10, activation="softmax")(xA)
outputsB = layers.Dense(10, activation="softmax")(xB)
modelC = keras.Model(inputs=inputs, outputs=[outputsA, outputsB])

print(modelB.summary())
plot_model(modelB, to_file='Sequential.png', show_shapes=False, show_dtype=False,
show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300)

print(modelC.summary())
plot_model(modelC, to_file='Functional.png', show_shapes=False, show_dtype=False,
show_layer_names=True, rankdir='TB', expand_nested=False, dpi=300)
