# BINARY CLASSIFIER VERSION 4
import datetime
import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Create the LeNet-type network

class LeNet:
   @staticmethod
   def build(width, height, depth, classes):
      # Initialise the model
      model = Sequential()
      inputShape = (height, width, depth)

      # Check if using channels first
      if K.image_data_format() == 'channels_first':
         inputShape = (depth, height, width)

      # First set of Convolutional > Relu > Pooling layers
      model.add(Conv2D(20, (5, 5), padding = "same", input_shape = inputShape))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

      # Second set of Convlolutional > Relu > Pooling layers
      model.add(Conv2D(50, (5, 5), padding = "same"))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

      # The only set of FC and RELU layers
      model.add(Flatten())
      model.add(Dense(500))
      model.add(Activation("relu"))

      # Softmax Classifier
      model.add(Dense(classes))
      model.add(Activation("softmax"))

      # Return the constructed network achitecture
      return model

# print(history.history.keys())
# # Summarize history for accuracy
# plt.figure(figsize = graph_size, dpi = 300)
# plt.grid(True, which = 'both')
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc = 'upper left')
# plt.savefig('Results/' + model_name + "-accuracy.png")
# plt.close()
#
# # Summarize history for loss
# plt.figure(figsize = graph_size, dpi = 300)
# plt.grid(True, which = 'both')
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc = 'upper left')
# plt.savefig('Results/' + model_name + "-loss.png")
# plt.close()
#
# # Serialise the model to JSON
# model_json = model.to_json()
# with open(model_name + ".json", "w") as json_file:
#    json_file.write(model_json)
#
# # Serialise weights to HDF5
# date_stamp = datetime.datetime.now().isoformat()
# weights_filename = str(model_name + '-' + str(date_stamp) + '.h5')
# model.save_weights(weights_filename)
# print("Runtime Complete. Model Saved to Disk.")
