# Using LeNet architecture

# Import packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
   @staticmethod
   def build(width, height, depth, classes):
      # Initialise the model
      model = Sequential()
      inputShape = (height, width, depth)

      # If 'channel first' is being used, update the input shape
      if K.image_data_format() == 'channel_first':
         inputShape = (depth, height, width)

      # Model Structure
      # First layer | CONV > RELU > POOL
      model.add(
         Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))  # Learning 20 (5 x 5) convolution filters
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

      # Second layer | CONV > RELU > POOL
      model.add(Conv2D(50, (5, 5), padding = "same"))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

      # Third layer | flattening out into fully-connected layers
      model.add(Flatten())
      model.add(Dense(500))  # 500 nodes
      model.add(Activation("relu"))

      # Softmax classifier
      model.add(Dense(classes))  # number of nodes = number of classes
      model.add(Activation("softmax"))  # yields probability for each class

      # Return the model
      return model