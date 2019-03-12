import datetime
import sys
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K, metrics
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

cat = cv2.imread('cat.jpg')


# plt.imshow(cat)
# plt.show()

# Build the network structure
def build_network_model(width, height, depth, classes):
    # Initialise the model
    model = Sequential()
    inputShape = (height, width, depth)

    # If 'channel first' is being used, update the input shape
    if K.image_data_format() == 'channel_first':
        inputShape = (depth, height, width)

    # Model Structure
    # First layer | CONV > RELU > POOL
    model.add(
        Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))  # Learning 20 (5 x 5) convolution filters
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Second layer | CONV > RELU > POOL
    model.add(Conv2D(50, (5, 5), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # Third layer | flattening out into fully-connected layers
    model.add(Flatten())
    model.add(Dense(50))  # 500 nodes
    model.add(Activation("relu"))

    # Softmax classifier
    model.add(Dense(classes))  # number of nodes = number of classes
    model.add(Activation("softmax"))  # yields probability for each class

    # Return the model
    return model


model = build_network_model(64, 64, 3, 2)

cat_batch = np.expand_dims(cat, axis = 0)
conv_cat = model.predict(cat_batch)


def visualise_cat(cat_batch):
    cat = np.squeeze(cat_batch, axis = 0)
    print(cat.shape)
    plt.imshow(cat)
    plt.show()


visualise_cat(conv_cat)
