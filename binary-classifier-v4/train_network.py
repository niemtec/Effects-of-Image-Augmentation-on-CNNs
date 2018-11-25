# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# Initialize the number of epochs to train for, initial learning rate, and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
modelName = 'classifier-v4'
plotName = modelName
datasetPath = 'datasets/cats-dogs/train'

# Initialize the data and labels
print("[INFO] loading images...")
sortedData = []
sortedLabels = []
data = []
labels = []


# Grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(datasetPath)))
# print("Image Paths:" + str(imagePaths))
# random.seed(42)
# random.shuffle(imagePaths)

# Image pre-processing
# Loop over the input images

def fileIsAnImage(path_to_file):
   filename, extension = os.path.splitext(path_to_file)
   if extension != '.jpg':
      return False
   else:
      return True


def shuffleArray(a, b):
   # Generate the permutation index
   permutation = np.random.permutation(a.shape[0])

   # Shuffle the array given the permutation
   shuffed_a = a[permutation]
   shuffled_b = b[permutation]
   return shuffed_a, shuffled_b


# Go through dataset directory
for datasetCategory in os.listdir(datasetPath):
   # Go through category 1 and then category 2 of the dataset
   datasetCategoryPath = datasetPath + "/" + datasetCategory
   for sample in os.listdir(datasetCategoryPath):
      if fileIsAnImage(datasetCategoryPath + "/" + sample) == True:
         print(sample)
         image = cv2.imread(datasetCategoryPath + "/" + sample)
         image = cv2.resize(image, (28, 28))  # Network only accepts 28 x 28 so resize the image accordingly
         image = img_to_array(image)
         sortedData.append(image)

         # Save label for the current image
         label = 1 if datasetCategory == 'cat' else 0
         sortedLabels.append(label)

combined = list(zip(sortedData, sortedLabels))
random.shuffle(combined)
data[:], labels[:] = zip(*combined)
print(labels)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)
# print(data)
# print(labels)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes = 2)
testY = to_categorical(testY, num_classes = 2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width = 28, height = 28, depth = 3, classes = 2)
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss = "binary_crossentropy", optimizer = opt,
              metrics = ["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BS),
                        validation_data = (testX, testY), steps_per_epoch = len(trainX) // BS,
                        epochs = EPOCHS, verbose = 1)

# save the model to disk
print("[INFO] serializing network...")
model.save(modelName)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig(plotName)
