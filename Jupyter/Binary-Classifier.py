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

resultsFileName = 'Demo'
datasetPath = 'Demo-dataset-rotation/'
rotationRange = 135
noEpochs = 100
initialLearningRate = 1e-5
batchSize = 32
decayRate = initialLearningRate / noEpochs
validationDatasetSize = 0.25
randomSeed = 42
imageDepth = 3

resultsPath = 'Demo-results/'
modelName = resultsFileName + "-" + str(rotationRange)
plotName = modelName

Tools = Helper(resultsPath, modelName)


# Build the network structure
def build_network_model(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)

	# If 'channel first' is being used, update the input shape
	if K.image_data_format() == 'channel_first':
		inputShape = (depth, height, width)

	# First layer
	model.add(
		Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))  # Learning 20 (5 x 5) convolution filters
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
for datasetCategory in os.listdir(datasetPath):
	datasetCategoryPath = datasetPath + "/" + datasetCategory

	# Go through category 1 and then category 2 of the dataset
	for sample in os.listdir(datasetCategoryPath):
		# print(stamp() + sample)
		if Tools.file_is_image(datasetCategoryPath + "/" + sample):
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
random_state = randomSeed)

# Convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes = 2)
testY = to_categorical(testY, num_classes = 2)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = rotationRange,fill_mode = "nearest")

augValidation = ImageDataGenerator(rotation_range = rotationRange,fill_mode = "nearest")

# Initialize the model
print(Tools.stamp() + "Compiling Network Model")

# Build the model based on control variable parameters
model = build_network_model(width = 64, height = 64, depth = imageDepth, classes = 3)

# Set optimiser
opt = Adam(lr = initialLearningRate, decay = decayRate)

# Compile the model using binary crossentropy, preset optimiser and selected metrics
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy", "mean_squared_error"])
# Train the network
print(Tools.stamp() + "Training Network Model")

# Save results of training in history dictionary for statistical analysis
history = model.fit_generator(
	aug.flow(trainX, trainY, batch_size = batchSize),
	validation_data = (testX, testY),
	steps_per_epoch = len(trainX) // batchSize,
	epochs = noEpochs,
	verbose = 1)

# Save all runtime statistics and plot graphs
Tools.save_network_stats(modelName, history, resultsFileName, 0, 0, 0)
Tools.save_accuracy_graph(history)
Tools.save_loss_graph(history)
Tools.save_model_to_disk(model)
Tools.save_weights_to_disk(model)

#TODO Add heatmap generation
