import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras

# The data has already been sorted into training and test sets for us
(train_data, train_labels),(test_data, test_labels) = fashion_mnist.load_data()

# Show the first training example
print(f"Training sample: \n{train_data[0]}\n")
print(f"Training label: \n{train_labels[0]}\n")

# Check the shape of a single example
train_data[0].shape, train_labels[0].shape

# plot a single sample
import matplotlib.pyplot as plt
plt.imshow(train_data[0])
plt.show()

# Check out samples label
train_labels[0]

# Create a small list so we can index onto training labels so they're human-readable
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

len(class_names)

# Plot an example image and its label
index_of_choice = 17
plt.imshow(train_data[index_of_choice], cmap=plt.cm.binary)
plt.title(class_names[train_labels[index_of_choice]])
plt.show()

# Plot multiple random images of fashion MNIST

import random

plt.figure(figsize=(7,7))
for i in range(4):
    ax = plt.subplot(2,2,i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
    
plt.show()

# Building a multi-class classification model

train_data[0].shape

len(class_names)

flatten_model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28))])
flatten_model.output_shape

28 * 28

train_labels[0]

# set the random seed
tf.random.set_seed(42)

# 1. Create the model using the sequential API
model_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])


# 2. Compile the model
model_1.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

# 3. Fit the model
non_norm_history = model_1.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

tf.one_hot(train_labels,depth=10)


# set the random seed
tf.random.set_seed(42)

# 1. Create the model using the sequential API
model_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])


# 2. Compile the model
model_1.compile(loss=keras.losses.CategoricalCrossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

# 3. Fit the model
non_norm_history_1 = model_1.fit(train_data, tf.one_hot(train_labels,depth=10), epochs=10, validation_data=(test_data, tf.one_hot(test_labels, depth=10)))



model_1.summary()


# Check the min and max values of the training data
train_data.min(), train_data.max()

# We can get our training and testing data between 0 & 1 by dividing by the maximum 
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# check the min and max values of the scaled training data
train_data_norm.min(), train_data_norm.max()

train_data_norm
train_data

# now our data is normalized, let's build a model to find patterns in it

# set the random seed
tf.random.set_seed(42)

# 1. Create the model using the sequential API
model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])


# 2. Compile the model
model_2.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

# 3. Fit the model
norm_history = model_2.fit(train_data_norm, train_labels, epochs=10, validation_data=(test_data_norm, test_labels))


# Comparing normalised and non-normalised data
import pandas as pd

# Plot non-normalized data loss curves
pd.DataFrame(non_norm_history.history).plot(title="Non-normalized data")

# Plot normalized data loss curves
pd.DataFrame(norm_history.history).plot(title="Normalized data")

plt.show()


# Finding the ideal learning rate

# set the random seed
tf.random.set_seed(42)

# 1. Create the model using the sequential API
model_3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])


# 2. Compile the model
model_3.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

# Create a learning rate callback
lr_schedular = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# 3. Fit the model
find_lr_history = model_3.fit(train_data_norm, 
                              train_labels, epochs=40, 
                              validation_data=(test_data_norm, test_labels),
                              callbacks=[lr_schedular])


# Plot the learning rate decay curve

import numpy as np
import matplotlib.pyplot as plt

lrs = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrs, find_lr_history.history["loss"])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")
plt.show()

10**-3

x = tf.constant([0.004], dtype=tf.float32)

log10_x = tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

print(log10_x.numpy())  # Output: [-3.]

# Let's refit a model with the ideal learning rate

# set the random seed
tf.random.set_seed(42)

# 1. Create the model using the sequential API
model_4 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])


# 2. Compile the model
model_4.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate = 0.001),
              metrics=["accuracy"])

# 3. Fit the model
history_4 = model_4.fit(train_data_norm, train_labels, epochs=20, validation_data=(test_data_norm, test_labels))




# Create a confusion matrix

import itertools
from sklearn.metrics import confusion_matrix

figsize = (10,10)

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size = 15):
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") /  cm.sum(axis=1)[:, np.newaxis] # normalize our confusion matrix
    n_classes = cm.shape[0]
    
    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    
    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    
    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    
    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    
    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i,j] > threshold else "black",
                 size=text_size)
    plt.show()   

class_names

# Make some predictions with our model
y_probs = model_4.predict(test_data) # probs is short for "prediction probabilities" 

# View the first 5 predictions
y_probs[:5]

test_data[0], test_data_norm[0]

# Make some predictions with our model
y_probs = model_4.predict(test_data_norm) # probs is short for "prediction probabilities" 

# View the first 5 predictions
y_probs[:5]

y_probs[0]
tf.argmax(y_probs[0])
class_names

class_names[tf.argmax(y_probs[0])]

class_names

y_probs[0], tf.argmax(y_probs[0]), class_names[tf.argmax(y_probs[0])]


# Convert all of the prediction probabilities into integers
y_preds = y_probs.argmax(axis=1)

# View the first 10 prediction labels
y_preds[:10]

test_labels

# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_labels, 
                      y_pred=y_preds, 
                      classes=class_names,
                      figsize=(15,15),
                      text_size=10)

# Visualising random model predictions

import random

def plot_random_image(model, images, true_labels, classes):
    """
    Picks a random image, plots it and labels it with a prediction and truth label.

    """
    
    # Set up random integer
    i = random.randint(0, len(images))
    
    # Create predictions and targets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label= classes[true_labels[i]]
    
    # Plot the image
    plt.imshow(target_image, cmap=plt.cm.binary)
    
    # Change the color of the titles depending on if the prediction is right or wrong
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
        
    # Add xlabel information (prediction/true label)
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100 * tf.reduce_max(pred_probs),
                                                     true_label),
               color = color) # Set the color to green or red based on if prediction is 

    
    plt.show()


# Check out a random image as well as its prediction
plot_random_image(model=model_4, 
                  images=test_data_norm, # Always make predictions on the same kind of data your model was trained on
                  true_labels=test_labels,
                  classes=class_names)

# What "patterns" is our model learning?

# Find the layers of our most recent model
model_4.layers


# Extract a particular layer
model_4.layers[1]

# Get the patterns of a layer in our network
weight, biases = model_4.layers[1].get_weights()

# Shapes
weight, weight.shape

28 * 28

model_4.summary()

# Biases and biases shape
biases, biases.shape


# Let's check out another way of viewing our deep learning models
from tensorflow.keras.utils import plot_model

# See the inputs and outputs of each layer
plot_model(model_4, show_shapes = True)

model_4.save("fashion_mnist_model.keras")

