from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt


# Control Variables
input_shape = (150, 150, 3)     # Input shape of the images (H x W x D)
nClasses = 2    # Number of classes for binary classification
batch_size = 540     # Number of samples to present to the network
epochs = 100    # Number of epochs to run for
training_directory = 'C://Users//janie//PycharmProjects//Project-Turing//train'
validation_directory = 'C://Users//janie//PycharmProjects//Project-Turing//validation'

# Defining the network model
def CreateModel():
    # Input Layer
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
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
    model.add(Flatten())    # This converts 3D feature maps to 1D feature vectors
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))    # Using softmax instead of sigmoid

    return model

# Load training dataset
train_datagen = ImageDataGenerator()
train_data = train_datagen.flow_from_directory(training_directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')

# Load validation dataset
validation_datagen = ImageDataGenerator
validation_data = validation_datagen.flow_from_directory(validation_directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')

# Build the network model
model1 = CreateModel()
# Using binary crossentropy loss for the model
model1.compile(optimiser='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the history of model fitting
history = model1.fit_generator(train_data, batch_size=batch_size, epochs=epochs, verbose = 1, validation_data=validation_data)

model1.evaluate_generator(validation_data)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

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