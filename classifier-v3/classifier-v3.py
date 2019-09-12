from keras.callbacks import Callback, LearningRateScheduler
from keras.initializers import glorot_uniform
from keras.layers import AveragePooling2D, BatchNormalization, Add
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD, Adadelta
from keras_applications.resnet50 import identity_block
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K, Model, Input
import numpy as np
import random
import classifier_helpers as tools

results_file_name = 'Batch-Size-2-test'
dataset_path = '../Cancer-Dataset/'
rotation_range = 360
epochs = 100
initial_learning_rate = 1e-5  # 1e-5
batch_size = 2
validation_dataset_size = 0.25
random_seed = 42
image_depth = 3

results_path = 'Results/'
model_name = results_file_name + "-" + str(rotation_range)
plot_name = model_name


def get_lr_metric(optimizer):
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
	model.add(
		Conv2D(20, (5, 5), padding = "same", input_shape = input_shape))  # Learning 20 (5 x 5) convolution filters
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


def identity_block(X, f, filters, stage, block):
	# Defining name basis
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	
	# Retrieve Filters
	F1, F2, F3 = filters
	
	# Save the input value
	X_shortcut = X
	
	# First component of main path
	X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2a',
	           kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
	X = Activation('relu')(X)
	
	# Second component of main path
	X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b',
	           kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
	X = Activation('relu')(X)
	
	# Third component of main path
	X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c',
	           kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
	
	# Final step: Add shortcut value to main path, and pass it through a RELU activation
	X = Add()([X, X_shortcut])
	X = Activation('relu')(X)
	
	return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape = (64, 64, 3), classes = 2):
	# Define the input as a tensor with shape input_shape
	X_input = Input(input_shape)
	
	# Zero-Padding
	X = ZeroPadding2D((3, 3))(X_input)
	
	# Stage 1
	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides = (2, 2))(X)
	
	# Stage 2
	X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1)
	X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'b')
	X = identity_block(X, 3, [64, 64, 256], stage = 2, block = 'c')
	
	# Stage 3
	X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2)
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
	X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')
	
	# Stage 4
	X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2)
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'b')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'c')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'd')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'e')
	X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'f')
	
	# Stage 5
	X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s = 2)
	X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'b')
	X = identity_block(X, 3, [512, 512, 2048], stage = 5, block = 'c')
	
	# AVGPOOL
	X = AveragePooling2D(pool_size = (2, 2), padding = 'same')(X)
	
	# Output layer
	X = Flatten()(X)
	X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes),
	          kernel_initializer = glorot_uniform(seed = 0))(X)
	
	# Create model
	model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
	
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


# lr_metric = get_lr_metric(optimiser)
opt = RMSprop(lr=initial_learning_rate, rho=0.9, epsilon=None, decay=initial_learning_rate/epochs)
opt2 = SGD(lr=initial_learning_rate, momentum=0.9, decay = initial_learning_rate/epochs, nesterov=False)
opt3 = Adam(initial_learning_rate)
opt4 = Adadelta(lr = initial_learning_rate, decay = initial_learning_rate/epochs)
opt5 = Adadelta(initial_learning_rate)
# Compile the model using binary crossentropy, preset optimiser and selected metrics
model.compile(loss = "binary_crossentropy", optimizer = opt3, metrics = ["accuracy", "mean_squared_error"])
# Train the network
print(tools.stamp() + "Training Network Model")


def stepDecay(epoch):
	dropEvery = 10
	initAlpha = 0.01
	factor = 0.25
	# Compute learning rate for current epoch
	exp = np.floor((1 + epoch) / dropEvery)
	alpha = initAlpha * (factor ** exp)
	
	return float(alpha)


# Save results of training in history dictionary for statistical analysis
history = model.fit_generator(
	training_augmented_image_generator.flow(train_x, train_y, batch_size = batch_size),
	validation_data = (test_x, test_y),
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
