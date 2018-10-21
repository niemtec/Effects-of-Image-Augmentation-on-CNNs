from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
history = model1.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose = 1, validation_data=(test_data, test_labels))

model1.evaluate(test_data, test_labels)