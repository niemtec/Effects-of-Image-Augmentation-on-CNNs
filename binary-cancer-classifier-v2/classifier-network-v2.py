from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import time

# Control Variables
input_shape = (150, 150, 3)  # Input shape of the images (H x W x D)
nClasses = 1  # Number of classes for binary classification
batch_size = 540  # Number of samples to present to the network
epochs = 100  # Number of epochs to run for
train_dir = 'C://Users//janie//PycharmProjects//Project-Turing//train'
val_dir = 'C://Users//janie//PycharmProjects//Project-Turing//validation'

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
    directory='train',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

# Load validation dataset
validation_generator = validation_datagen.flow_from_directory(
    directory='validation',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

# Save the history of model fitting
history = model.fit_generator(
    train_generator,
    steps_per_epoch=1000 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

model.evaluate_generator(validation_generator)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

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
