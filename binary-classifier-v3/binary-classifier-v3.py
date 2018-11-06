# BINARY CLASSIFIER VERSION 3
import datetime

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

########################################################################################################################
# CONTROL VARIABLES
batch_size = 30
number_of_epochs = 50
number_of_samples = None
number_of_steps_per_epoch = 2000 // batch_size
number_of_validation_steps = 800 // batch_size
training_directory = '../datasets/cats-dogs/train'
validation_directory = '../datasets/cats-dogs/validation'
model_name = 'binary-classifier-v3'
########################################################################################################################

# Build the sequential convolutional model for image classification
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), data_format = 'channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# First convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# The model outputs 3D feature maps (height, width, features)
# Fully connected layers
model.add(Flatten())  # This converts 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))  # Sigmoid activation function

# Using binary crossentropy loss for the model
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# This is the augmentation configuration used for training
train_datagen = ImageDataGenerator(
   rescale = 1. / 255,
   horizontal_flip = True,
   vertical_flip = True,
   zoom_range = 0.2,
   rotation_range = 90,
   fill_mode = 'nearest')

# This is the augmentation configuration used for testing:
test_datagen = ImageDataGenerator(rescale = 1. / 255)

# These generators will read pictures found in sub-folders and generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
   training_directory,  # this is the target directory
   target_size = (150, 150),  # all images will be resized to 150x150
   batch_size = batch_size,
   class_mode = 'binary')  # since we use binary_crossentropy loss, we need binary labels

validation_generator = test_datagen.flow_from_directory(
   validation_directory,
   target_size = (150, 150),
   batch_size = batch_size,
   class_mode = 'binary')

history = model.fit_generator(
   train_generator,
   steps_per_epoch = number_of_steps_per_epoch,
   epochs = number_of_epochs,
   validation_data = validation_generator,
   validation_steps = number_of_validation_steps)

# Serialise the model to JSON
model_json = model.to_json()
with open(model_name + ".json", "w") as json_file:
   json_file.write(model_json)

# Serialise weights to HDF5
date_stamp = datetime.datetime.now()
model.save_weights(model_name + "-" + str(date_stamp) + ".h5")
print("Runtime Complete. Model Saved to Disk.")

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig(model_name + "-accuracy.png")

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig(model_name + "-loss.png")
