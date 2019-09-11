from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
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
import classifier_helpers as tools

results_file_name = 'Batch-Size-000'
dataset_path = '../Cancer-Dataset/'
rotation_range = 0
epochs = 10
initial_learning_rate = 1e-5
batch_size = 2
decay_rate = initial_learning_rate / epochs  # TODO: Determine the manual decay rate
validation_dataset_size = 0.25
random_seed = 42
image_depth = 3

results_path = 'Results/'
model_name = results_file_name + "-" + str(rotation_range)
plot_name = model_name


def get_learning_rate_metric(optimizer):
	def lr(y_true, y_pred):
		return optimizer.lr
	
	return lr


# Build the network structure
def buildNetworkModel(width, height, depth, classes):
	model = Sequential()
	input_shape = (height, width, depth)
	
	# If 'channel first' is being used, update the input shape
	if K.image_data_format() == 'channel_first':
		input_shape = (depth, height, width)
	
	# First layer
	model.add(Conv2D(20, (5, 5), padding = "same", input_shape = input_shape))  # Learning 20 (5 x 5) convolution filters
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

# Load the dataset
sorted_data = np.load('sorted_data_array.npy')
sorted_labels = np.load('sorted_labels_array.npy')
data = []
labels = []

combined = list(zip(sorted_data, sorted_labels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

test_set = int(validation_dataset_size * len(labels))
validation_dataset_labels = labels[-test_set:]

# Partition the data into training and testing splits
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size = test_set, random_state = random_seed)

# Convert the labels from integers to vectors
train_y = to_categorical(train_y, num_classes = 2)
test_y = to_categorical(test_y, num_classes = 2)

# Construct the image generator for data augmentation
training_augmented_image_generator = ImageDataGenerator(rotation_range = rotation_range, fill_mode = "nearest")
testing_augmented_image_generator = ImageDataGenerator(rotation_range = rotation_range, fill_mode = "nearest")

# Initialize the model
print(tools.stamp() + "Compiling Network Model")

# Build the model based on control variable parameters
model = buildNetworkModel(width = 64, height = 64, depth = image_depth, classes = 2)

# Set optimiser
optimiser = Adam(lr = initial_learning_rate, decay = decay_rate)

lr_metric = get_learning_rate_metric(optimiser)

# Compile the model using binary crossentropy, preset optimiser and selected metrics
model.compile(loss = "binary_crossentropy", optimizer = optimiser, metrics = ["accuracy", "mean_squared_error", lr_metric])
# Train the network
print(tools.stamp() + "Training Network Model")

# Save results of training in history dictionary for statistical analysis
history = model.fit_generator(
	training_augmented_image_generator.flow(train_x, train_y, batch_size = batch_size),
	validation_data = (test_x, test_y),
	steps_per_epoch = len(train_x) // batch_size,
	epochs = epochs,
	verbose = 2)

# Save all runtime statistics and plot graphs
tools.saveNetworkStats(history, epochs, initial_learning_rate, model_name, results_path)
tools.saveAccuracyGraph(history, plot_name, results_path)
tools.saveLossGraph(history, plot_name, results_path)
tools.saveLearningRateGraph(history, plot_name, results_path)
tools.saveModelToDisk(model, model_name, results_path)
tools.saveWeightsToDisk(model, model_name, results_path)

# TODO Add heatmap generation
