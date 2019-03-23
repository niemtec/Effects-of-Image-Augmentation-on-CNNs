import datetime
import sys
import matplotlib
from keras.engine.saving import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K, metrics
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import cv2

plt.interactive(False)

sample = cv2.imread('sample.jpg')


# sample = plt.imread('sample.jpg')
# plt.imshow(sample)
# plt.show()

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("weights.h5")
    print("Loaded model from disk")

    return loaded_model


def nice_priner(model, image):
    image_batch = np.expand_dims(image, axis = 0)
    conv_image2 = model.predict(image_batch)

    conv_image2 = np.squeeze(conv_image2, axis = 0)
    print(conv_image2.shape)
    conv_image2 = conv_image2.reshape(conv_image2.shape[:2])

    print(conv_image2.shape)
    plt.imshow(conv_image2)
    plt.show()


model = load_model()

nice_priner(model, sample)
