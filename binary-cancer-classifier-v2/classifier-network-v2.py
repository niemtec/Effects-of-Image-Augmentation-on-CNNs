from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


input_shape = (150, 150, 3)
nClasses = 2
batch_size = 30
epochs = 100
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
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax')) # Using softmax instead of sigmoid

    return model

train_datagen = ImageDataGenerator()

train_data = train_datagen.flow_from_directory(training_directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')

validation_datagen = ImageDataGenerator

validation_data = validation_datagen.flow_from_directory(validation_directory, target_size=(150, 150), batch_size=batch_size, class_mode='binary')


model1 = CreateModel()

model1.compile(optimiser='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose = 1, validation_data=(test_data, test_labels))

model1.evaluate(test_data, test_labels)