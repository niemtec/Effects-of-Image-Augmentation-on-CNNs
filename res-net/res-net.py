import os
import random
from binary_classifier import get_lr_metric
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.layers import BatchNormalization, Activation, Conv2D, AveragePooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.python.estimator import keras
import classifier_helpers as tools
import numpy as np

results_file_name = 'Batch-Size-2'
dataset_path = '../Cancer-Dataset/'
rotation_range = 0
epochs = 100
initial_learning_rate = 1e-5  # 1e-5
batch_size = 2
decay_rate = initial_learning_rate / epochs  # TODO: Determine the manual decay rate
validation_dataset_size = 0.25
random_seed = 42
image_depth = 3
results_path = 'Results/'
model_name = results_file_name + "-" + str(rotation_range)
plot_name = model_name


def lr_schedule(epoch):
	"""Learning Rate Schedule

	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.

	# Arguments
		epoch (int): The number of epochs

	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 80:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


def resnet_layer(inputs,
                 num_filters = 16,
                 kernel_size = 3,
                 strides = 1,
                 activation = 'relu',
                 batch_normalization = True,
                 conv_first = True):
	"""2D Convolution-Batch Normalization-Activation stack builder

	# Arguments
		inputs (tensor): input tensor from input image or previous layer
		num_filters (int): Conv2D number of filters
		kernel_size (int): Conv2D square kernel dimensions
		strides (int): Conv2D square stride dimensions
		activation (string): activation name
		batch_normalization (bool): whether to include batch normalization
		conv_first (bool): conv-bn-activation (True) or
			bn-activation-conv (False)

	# Returns
		x (tensor): tensor as input to the next layer
	"""
	conv = Conv2D(num_filters,
	              kernel_size = kernel_size,
	              strides = strides,
	              padding = 'same',
	              kernel_initializer = 'he_normal',
	              kernel_regularizer = l2(1e-4))
	
	x = inputs
	if conv_first:
		x = conv(x)
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
	else:
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		x = conv(x)
	return x


def buildNetworkModel(width, height, depth, num_classes):
	input_shape = (height, width, depth)
	
	if (depth - 2) % 6 != 0:
		raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
	# Start model definition.
	num_filters = 16
	num_res_blocks = int((depth - 2) / 6)
	
	inputs = Input(shape = input_shape)
	x = resnet_layer(inputs = inputs)
	# Instantiate the stack of residual units
	for stack in range(3):
		for res_block in range(num_res_blocks):
			strides = 1
			if stack > 0 and res_block == 0:  # first layer but not first stack
				strides = 2  # downsample
			y = resnet_layer(inputs = x,
			                 num_filters = num_filters,
			                 strides = strides)
			y = resnet_layer(inputs = y,
			                 num_filters = num_filters,
			                 activation = None)
			if stack > 0 and res_block == 0:  # first layer but not first stack
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs = x,
				                 num_filters = num_filters,
				                 kernel_size = 1,
				                 strides = strides,
				                 activation = None,
				                 batch_normalization = False)
			x = keras.layers.add([x, y])
			x = Activation('relu')(x)
		num_filters *= 2
	
	# Add classifier on top.
	# v1 does not use BN after last shortcut connection-ReLU
	x = AveragePooling2D(pool_size = 8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
	                activation = 'softmax',
	                kernel_initializer = 'he_normal')(y)
	
	# Instantiate model.
	model = Model(inputs = inputs, outputs = outputs)
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
model = buildNetworkModel(height = 64, width = 64, depth = image_depth, num_classes = 2)
optimizer = Adam(lr = lr_schedule(0))
lr_metric = get_lr_metric(optimizer)
model.compile(loss = 'binary_crossentropy', optimizer = optimizer,
              metrics = ["accuracy", "mean_squared_error", lr_metric])

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % "resnet"
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath = filepath,
                             monitor = 'val_acc',
                             verbose = 1,
                             save_best_only = True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1),
                               cooldown = 0,
                               patience = 5,
                               min_lr = 0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Train the network
print(tools.stamp() + "Training Network Model")

history = model.fit_generator(training_augmented_image_generator.flow(train_x, train_y, batch_size = batch_size),
                              validation_data = (test_x, test_y),
                              steps_per_epoch = len(train_x) // batch_size,
                              epochs = epochs,
                              verbose = 2,
                              workers = 4,
                              callbacks = callbacks)

# Save all runtime statistics and plot graphs
tools.saveNetworkStats(history, epochs, initial_learning_rate, model_name, results_path)
tools.saveAccuracyGraph(history, plot_name, results_path)
tools.saveLossGraph(history, plot_name, results_path)
tools.saveLearningRateGraph(history, plot_name, results_path)
tools.saveModelToDisk(model, model_name, results_path)
tools.saveWeightsToDisk(model, model_name, results_path)
