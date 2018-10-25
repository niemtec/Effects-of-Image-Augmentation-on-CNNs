from keras.backend import set_session, tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import time

# Control Variables
input_shape = (150, 150, 3)  # Input shape of the images (H x W x D)
target_size = (150, 150)
nClasses = 1  # Number of classes for binary classification
batch_size = 10  # Number of samples to present to the network
epochs = 100  # Number of epochs to run for
number_of_samples = 540
number_of_evaluation_samples = 540
steps_per_epoch = None # None = default = number of samples / batch size
validation_steps = None
training_directory = 'train'
validation_directory = 'validation'


# Defining the network model
# Input Layer
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, data_format='channels_last'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# First convolutional layer
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second convolutional layer
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())  # This converts 3D feature maps to 1D feature vectors
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#model.add(Dense(nClasses, activation='softmax'))  # Using softmax instead of sigmoid

# Using binary crossentropy loss for the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    directory=training_directory,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

# Load validation dataset
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_directory,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

# Save the history of model fitting
history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

evalhistory = model.evaluate_generator(validation_generator, steps=number_of_evaluation_samples // batch_size)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Serialise the model to JSON
model_json = model.to_json()
with open('classifier-network-v2.json', "w") as json_file:
    json_file.write(model_json)

# Serialise weights to HDF5
datestamp = time.time()
model.save_weights(str(datestamp) + ".h5")  # always save your weights after training or during training
print("Runtime Complete. Model Saved to Disk.")

