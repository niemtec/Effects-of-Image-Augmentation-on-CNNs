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
import numpy as np
import random
import cv2
import os
from classifierHelpers import Helper

results_file_name = 'Demo'
dataset_path = 'Demo-dataset-rotation/'
rotation_range = 135
epochs = 100
initial_learning_rate = 1e-5
batch_size = 32
decay_rate = initial_learning_rate / epochs
validation_dataset_size = 0.25
random_seed = 42
image_depth = 3

results_path = 'Demo-results/'
model_name = results_file_name + "-" + str(rotation_range)
plot_name = model_name

Tools = Helper(results_path, model_name)


# Build the network structure
def buildNetworkModel(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)

	# If 'channel first' is being used, update the input shape
	if K.image_data_format() == 'channel_first':
		inputShape = (depth, height, width)

	# First layer
	model.add(Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))  # Learning 20 (5 x 5) convolution filters
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

	# Second layer
	model.add(Conv2D(50, (5, 5), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

	# Third layer - fully-connected layers
	model.add(Flatten())
	model.add(Dense(50))  # 500 nodes
	model.add(Activation("relu"))

	# Softmax classifier
	model.add(Dense(classes))  # number of nodes = number of classes
	model.add(Activation("softmax"))  # yields probability for each class

	# Return the model
	return model


# Initialize the data and labels arrays
sortedData = []
sortedLabels = []
data = []
labels = []

# Go through dataset directory
print(Tools.stamp() + "Classifying the Dataset")
for datasetCategory in os.listdir(dataset_path):
	datasetCategoryPath = dataset_path + "/" + datasetCategory

	# Go through category 1 and then category 2 of the dataset
	for sample in os.listdir(datasetCategoryPath):
		# print(stamp() + sample)
		if Tools.isFileAnImage(datasetCategoryPath + "/" + sample):
			image = cv2.imread(datasetCategoryPath + "/" + sample)
			image = cv2.resize(image, (
				64, 64))
			image = img_to_array(image)
			# Save image to the data list
			sortedData.append(image)

			# Decide on binary label
			if datasetCategory == 'benign':
				label = 1
			else:
				label = 0
			# Save label for the current image
			sortedLabels.append(label)

combined = list(zip(sortedData, sortedLabels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

validationDatasetLabels = []
#TODO: Fix Demo settings of 7
# testSet = 0.25 * len(labels)
validationDatasetLabels = labels[-7:]
#TODO: Fix Demo settings of 7
# Partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 7,
                                                  random_state = random_seed)

# Convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes = 2)
testY = to_categorical(testY, num_classes = 2)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = rotation_range, fill_mode = "nearest")

augValidation = ImageDataGenerator(rotation_range = rotation_range, fill_mode = "nearest")

# Initialize the model
print(Tools.stamp() + "Compiling Network Model")

# Build the model based on control variable parameters
model = buildNetworkModel(width = 64, height = 64, depth = image_depth, classes = 3)

# Set optimiser
opt = Adam(lr = initial_learning_rate, decay = decay_rate)

# Compile the model using binary crossentropy, preset optimiser and selected metrics
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy", "mean_squared_error"])
# Train the network
print(Tools.stamp() + "Training Network Model")

# Save results of training in history dictionary for statistical analysis
history = model.fit_generator(
	aug.flow(trainX, trainY, batch_size = batch_size),
	validation_data = (testX, testY),
	steps_per_epoch = len(trainX) // batch_size,
	epochs = epochs,
	verbose = 1)

# Save all runtime statistics and plot graphs
Tools.saveNetworkStats(model_name, history, results_file_name, 0, 0, 0)
Tools.saveAccuracyGraph(history)
Tools.saveLossGraph(history)
Tools.saveModelToDisk(model)
Tools.saveWeightsToDisk(model)

#TODO Add heatmap generation
