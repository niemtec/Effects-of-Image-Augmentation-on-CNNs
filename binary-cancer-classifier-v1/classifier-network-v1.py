import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import time

# Network configuration to use GPUs
config = tf.ConfigProto(
   log_device_placement = True  # Show which device is being used
   # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
   # device_count = {'GPU': 1, 'CPU:': 8}
)
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
set_session(session)

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

# Data Preparation
batch_size = 30

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale = 1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale = 1. / 255)

# this is a generator that will read pictures found in
# sub-folers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
   '../datasets/isic/train',  # this is the target directory
   target_size = (150, 150),  # all images will be resized to 150x150
   batch_size = batch_size,
   class_mode = 'binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
   '../datasets/isic/validation',
   target_size = (150, 150),
   batch_size = batch_size,
   class_mode = 'binary')

history = model.fit_generator(
   train_generator,
   steps_per_epoch = 1000 // batch_size,
   epochs = 100,
   validation_data = validation_generator,
   validation_steps = 800 // batch_size)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('results/100 Epochs/accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.savefig('results/100 Epochs/loss.png')

# Serialise the model to JSON
model_json = model.to_json()
with open("results/100 Epochs/classifier-network-v1.json", "w") as json_file:
   json_file.write(model_json)

# Serialise weights to HDF5
datestamp = time.time()
model.save_weights(
   "results/100 Epochs" + str(datestamp) + ".h5")  # always save your weights after training or during training
print("Runtime Complete. Model Saved to Disk.")
