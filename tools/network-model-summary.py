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
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

# BINARY CLASSIFIER NETWORK BASED ON THE LENET MODEL BUILT FOR IMAGE CLASSIFICATION TASKS
# PART OF 'PROJECT TURING' - JAKUB ADRIAN NIEMIEC (@niemtec)
# THIS MODEL IS USED AS A TOOL FOR INVESTIGATING THE PHENOMENA OF OVERFITTING IN CONVOLUTIONAL NEURAL NETWORKS
# Set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

# Control Variables
noEpochs = 100
initialLearningRate = 1e-5
batchSize = 32
decayRate = initialLearningRate / noEpochs
numberOfClasses = 2
validationDatasetSize = 0.25  # Using 75% of the data for training and the remaining 25% for testing
randomSeed = 42  # For repeatability
imageHeight = 64
imageWidth = 64
imageDepth = 3


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


# Construct the image generator for data augmentation
aug = ImageDataGenerator(
    # rotation_range = 25
    # vertical_flip = True
    # horizontal_flip= True
    # zoom_range = 1.0
    # width_shift_range = 0.1
    # height_shift_range = 0.1,
    # shear_range = 0.2,
    # fill_mode = "nearest"
)

# Initialize the model
model = build_network_model(width = imageWidth, height = imageHeight, depth = imageDepth, classes = numberOfClasses)
opt = Adam(lr = initialLearningRate, decay = decayRate)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

model.summary()
