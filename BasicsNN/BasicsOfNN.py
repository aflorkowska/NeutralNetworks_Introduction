# Ignoring printing warnings and info. Still showing errors.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import backend as K

import plotMethods

# Using GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Data loading, splitting and statistics
print(f"---LOADING---")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=0, train_size = .5)
print(f"Training data size: {x_train.shape[0]}")
print(f"Validation data size: {x_val.shape[0]}")
print(f"Testing data size: {x_test.shape[0]}")
print(f"Single input data dimension (= image size): {x_test.shape[1]} x {x_test.shape[2]}")
num_classes = np.size(np.unique(y_train))
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(f"Number of classes {num_classes} with labels {labels}")

#trainDistribution = plotMethods.bar(np.unique(y_train, return_counts=True)[0], np.unique(y_train, return_counts=True)[1], "Distribution of the numbers training data of various classes")
#trainDistribution.show()
#valDistribution = plotMethods.bar(np.unique(y_val, return_counts=True)[0], np.unique(y_test, return_counts=True)[1], "Distribution of the numbers validation data of various classes")
#valDistribution.show()
#testDistribution = plotMethods.bar(np.unique(y_test, return_counts=True)[0], np.unique(y_test, return_counts=True)[1], "Distribution of the numbers testing data of various classes")
#testDistribution.show()

#trainSetSamples = plotMethods.samplesOfSet(x_train, y_train, "Training set - examples", 2, 5, 0, 10)
#trainSetSamples.show()
#valSetSamples = plotMethods.samplesOfSet(x_val, y_val, "Validation set - examples", 2, 5, 0, 10)
#valSetSamples.show()
#testSetSamples = plotMethods.samplesOfSet(x_test, y_test, "Testing set - examples", 2, 5, 0, 10)
#testSetSamples.show()
print(f"")

# Preprocessing
print(f"---PREPROCESSING---")
print(f"BEFORE preprocessing: training data size: {x_train.shape}, validation data size: {x_val.shape} and testing data size: {x_test.shape}")
x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2]).astype("float32") / 255.0
x_val = x_val.reshape(-1, x_val.shape[1] * x_val.shape[2]).astype("float32") / 255.0
x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2]).astype("float32") / 255.0
print(f"AFTER preprocessing: training data size: {x_train.shape}, validation data size: {x_val.shape} and testing data size: {x_test.shape}")

#y_train = tf.keras.utils.to_categorical(y_train, num_classes)
#y_val = tf.keras.utils.to_categorical(y_val, num_classes)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print(f"")

# Model
print(f"---MODEL---")
inputs = keras.Input(shape=(x_train.shape[1]))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["acc"],
)
print(model.summary())
print(f"")

# Training
print(f"---TRAINING---")
history = model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2, validation_data=(x_val, y_val))
print(f"")

# Evaluation
print(f"---EVALUATION---")
train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=32, verbose=2)
print(f"On training data model obtained accuracy equals to {train_acc:.2f} wit loss {train_loss:.2f}. It means that bias is {train_acc:.2f}, assuming that the Bayes optimal error is equal to 0.")

val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=32, verbose=2)
print(f"On validation data model obtained accuracy equals to {val_acc:.2f} wit loss {val_loss:.2f}. It means that variance is {(val_acc - train_acc):.2f},")

#accuracyComparison = plotMethods.compareResults(history.history['acc'], history.history['val_acc'], "Model accuracy", "Epoch", "Accuracy")
#accuracyComparison.show()
#lossComparison = plotMethods.compareResults(history.history['loss'], history.history['val_loss'], "Model loss", "Epoch", "Loss")
#lossComparison.show()

argmax = np.argmax(history.history['val_acc'])
acc_max = history.history['val_acc'][argmax]
print(f'The best accuracy {acc_max:.2f} for validation set we achieved in epoch: {argmax:.2f}')

argmin = np.argmin(history.history['val_loss'])
loss_min = history.history['val_loss'][argmin]
print(f'Overfitting stated after epoch:  {argmin:.2f} where the minimum loss was: {loss_min:.2f}')

print(f"")

#Testing
print(f"---TESTING---")
y_test_prediction = model.predict(x_test)
y_test_predictionMAX = np.argmax(y_test_prediction, axis = 1) # taking the class with the highest probability
confusionMatrix = plotMethods.show_confusion_matrix(y_test, y_test_predictionMAX, num_classes, labels)
#confusionMatrix.show()

print(classification_report(y_test, y_test_predictionMAX))
print(f"")

# # Finding out misclassified examples
classcheck = y_test - y_test_predictionMAX # 0 when the class is the same, otherwise example is misclassified
misclasified = np.where(classcheck != 0)[0]
num_misclassified = len(misclasified)
# Verify if it works correctly
# Refractor method
misclassifiedSetSamples = plotMethods.misclassificationReport(x_test, misclasified, y_test, y_test_predictionMAX, "Misclassification report in format [true label : prediction]", 2, 5, 0, num_misclassified)
misclassifiedSetSamples.show()


###### TODO
### Basic part
# Labels encoding
# ROC curves
# DataLoaders
# Data augmentation
# Regularizations
# Types of layers
# Dropout
# Callbacks
# Tensorflow board
# Saving models and loading them

### Advanced, useful and detailed
# Custom training loops
# Custom dataset
# Custom layers
# Custom model.fit method

