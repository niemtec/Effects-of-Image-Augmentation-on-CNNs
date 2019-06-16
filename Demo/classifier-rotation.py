#!/usr/bin/env python

"""
Experiment setup script for investigating the phenomena of overfitting in Convolutional Neural Networks
using binary classification of cancer images. Model based on a modified LeNet architecture focussed on 
binary classification of samples.

This project has been put together as part of a dissertation in BSc (Hons) Computer Science with Artificial Intelligence.
"""

__author__ = "Jakub Adrian Niemiec"


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
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
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os


# Control Variables
# Name of the experiment performed (used for graph titles etc.)
datasetName = 'all-corrupted'
resultsFileName = 'Demo'  # Name of the result files
rotationRange = 135  # Range of rotations for the current experiment eg. 0, 45, 90, 135, 180
categoryOne = 'benign'  # Name of the first category for classification
categoryTwo = 'malignant'   # Name of the second category for classification
# Name of the model used throughout graphs and results
modelName = datasetName + "-" + str(rotationRange)
# Path to the dataset to be used for classification
datasetPath = 'Demo-dataset-rotation/'
resultsPath = 'Demo-results/'   # Path to be used for output of graphs and statistics
plotName = modelName    # Name of the graph (using model name)
graphSize = (15, 10)  # Size of result plots
noEpochs = 100   # Number of epochs to run the model for (default 100)
# Learning rate (determined from previous experimentation on dataset)
initialLearningRate = 1e-5
batchSize = 32  # Size of sample batches to feed to the network
decayRate = initialLearningRate / noEpochs
# Number of classification classes (two for binary classification)
numberOfClasses = 2
# Using 75% of the data for training and the remaining 25% for testing
validationDatasetSize = 0.25
randomSeed = 42  # For repeatability
imageHeight = 64    # Input image height
imageWidth = 64     # Input image width
# Input image depth (three means Red, Green, Blue coloured image)
imageDepth = 3

# Determine whether given file is an image


def file_is_image(path_to_file):
    filename, extension = os.path.splitext(path_to_file)
    if extension != '.jpg':
        return False
    else:
        return True


# Prints current timestamp, to be used in print statements
def stamp():
    time = "[" + str(datetime.datetime.now().time()) + "]   "
    return time


# Save final model performance
def save_network_stats(resultsPath, modelName, history, fileName, sensitivity, specificity, precision):
    # Extract data from history dictionary
    historyLoss = history.history['loss']
    historyLoss = str(historyLoss[-1])  # Get last value from loss
    historyAcc = history.history['acc']
    historyAcc = str(historyAcc[-1])  # Get last value from accuracy
    historyValLoss = history.history['val_loss']
    # Get last value from validated loss
    historyValLoss = str(historyValLoss[-1])
    historyValAcc = history.history['val_acc']
    # Get last value from validated accuracy
    historyValAcc = str(historyValAcc[-1])
    historyMSE = 0  # str(historyMSE[-1])
    historyMAPE = 0  # history.history['mape']
    historyMAPE = 0  # str(historyMAPE[-1])

    with open(resultsPath + fileName + ".txt", "a") as history_log:
        history_log.write(
            modelName + "," + historyLoss + "," + historyAcc + "," + historyValLoss + "," + historyValAcc + "," + str(
                noEpochs) + "," + str(initialLearningRate) + "," + str(historyMSE) + "," + str(
                historyMAPE) + "," + str(sensitivity) + "," + str(specificity) + "," + str(precision) + "\n")
    history_log.close()

    print(stamp() + "Keras Log Saved")

    print(history.history.keys())

    print(stamp() + "History File Saved")


# Build the network structure
def build_network_model(width, height, depth, classes):
    # Initialise the model
    model = Sequential()
    inputShape = (height, width, depth)

    # If 'channel first' is being used, update the input shape
    if K.image_data_format() == 'channel_first':
        inputShape = (depth, height, width)

    # First layer
    model.add(
        Conv2D(20, (5, 5), padding="same", input_shape=inputShape))  # Learning 20 (5 x 5) convolution filters
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second layer
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Third layer - fully-connected layers
    model.add(Flatten())
    model.add(Dense(50))  # 500 nodes
    model.add(Activation("relu"))

    # Softmax classifier
    model.add(Dense(classes))  # number of nodes = number of classes
    model.add(Activation("softmax"))  # yields probability for each class

    # Return the model
    return model


# Calculate confusion matrix statistics
def calculate_statistics(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    precision = tp / (tp + fp)

    return sensitivity, specificity, precision


# Save the confusion matrix as a graphical figure
def save_confusion_matrix(tp, tn, fp, fn):
    import seaborn as sns
    tp = int(tp)
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)

    cm = [[tp, tn], [fp, fn]]
    cm = np.array(cm)
    heatmap = sns.heatmap(cm, annot=True, fmt='g', linewidths=0.2)
    fig = heatmap.get_figure()
    fig.savefig(resultsPath + '/' + modelName + '-confusion-matrix.png')


# Summarize history for accuracy
def save_accuracy_graph(history):
    plt.figure(figsize=graphSize, dpi=75)
    plt.grid(True, which='both')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.suptitle(modelName)
    plt.savefig(resultsPath + '/' + modelName + "-accuracy.png")
    plt.close()


# Summarize history for loss
def save_loss_graph(history):
    plt.figure(figsize=graphSize, dpi=75)
    plt.grid(True, which='both')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.suptitle(modelName)
    plt.savefig(resultsPath + '/' + modelName + "-loss.png")
    plt.close()


# Initialize the data and labels arrays
sortedData = []
sortedLabels = []
data = []
labels = []

# Go through dataset directory
print(stamp() + "Classifying the Dataset")
for datasetCategory in os.listdir(datasetPath):
    datasetCategoryPath = datasetPath + "/" + datasetCategory

    # Go through category 1 and then category 2 of the dataset
    for sample in os.listdir(datasetCategoryPath):
        # print(stamp() + sample)
        if file_is_image(datasetCategoryPath + "/" + sample):
            image = cv2.imread(datasetCategoryPath + "/" + sample)
            image = cv2.resize(image, (
                imageHeight, imageWidth))  # Network only accepts 28 x 28 so resize the image accordingly
            image = img_to_array(image)
            # Save image to the data list
            sortedData.append(image)

            # Decide on binary label
            if datasetCategory == categoryOne:
                label = 1
            else:
                label = 0
            # Save label for the current image
            sortedLabels.append(label)

combined = list(zip(sortedData, sortedLabels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

validationDatasetLabels = []
# testSet = 0.25 * len(labels)
validationDatasetLabels = labels[-7:]

# Partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=7,  # Set manually to 7 for Demo
                                                  random_state=randomSeed)

# Convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=numberOfClasses)
testY = to_categorical(testY, num_classes=numberOfClasses)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=rotationRange,
    fill_mode="nearest"
)

augValidation = ImageDataGenerator(
    rotation_range=rotationRange,
    fill_mode="nearest"
)

# Initialize the model
print(stamp() + "Compiling Network Model")

# Build the model based on control variable parameters
model = build_network_model(
    width=imageWidth, height=imageHeight, depth=imageDepth, classes=numberOfClasses)

# Set optimiser
opt = Adam(lr=initialLearningRate, decay=decayRate)

# Compile the model using binary crossentropy, preset optimiser and selected metrics
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy", "mean_squared_error", "mean_absolute_error"])
# Train the network
print(stamp() + "Training Network Model")

# Save results of training in history dictionary for statistical analysis
history = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=batchSize),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batchSize,
    epochs=noEpochs,
    verbose=1)

# The following can be used to produce confusion matrices if necessary
# predictions = model.predict_classes(testX, batchSize, 0)
# tn, fp, fn, tp = confusion_matrix(validationDatasetLabels, predictions).ravel()
# print(tn, fp, fn, tp)
sensitivity, specificity, precision = 0, 0, 0   # Set to 0 if not used
# sensitivity, specificity, precision = calculate_statistics(tn, fp, fn, tp)

# Save all runtime statistics and plot graphs
save_network_stats(resultsPath, modelName, history,
                   resultsFileName, sensitivity, specificity, precision)
# save_confusion_matrix(tn, fp, fn, tp)
save_accuracy_graph(history)
save_loss_graph(history)

# Save the model to disk
print(stamp() + "Saving Network Model")
model_json = model.to_json()
with open(resultsPath + '/' + modelName + ".json", "w") as json_file:
    json_file.write(model_json)

# Save weights to disk
print(stamp() + "Saving Network Weights")
model.save_weights(resultsPath + '/' + modelName + ".h5", "w")
