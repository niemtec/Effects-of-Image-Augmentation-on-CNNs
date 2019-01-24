# Tool used to print the dataset images for pictographic representation of data augmentation techniques

from keras.preprocessing.image import img_to_array
import random
import cv2
import math
import os
import matplotlib.pyplot as plt


# Resize the image to desired output
# def resize_image(image):
#     image = cv2.resize(image, (imageHeight, imageWidth))
#     return image


# Method for loading the image from path
def convert_image_to_array(image):
    # Convert image to array
    imageArray = img_to_array(image)
    return imageArray


# Save the image as a file
def save_image(image, newImagePath):
    # Save the image with the BGR (default cv2) encoding to avoid colour shift
    # cv2.imwrite(newImagePath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(newImagePath, image)

# Pick image co-ordinates to alter
def pick_alteration_coordinates(width, height):
    # Pick random co-ordinates to change
    xAxisChange = random.randint(0, (width - 1))
    yAxisChange = random.randint(0, (height - 1))
    return xAxisChange, yAxisChange


# Alter the image by turning off pixels at random
def alter_image(image, augmentationFactor):
    # Get image size
    imageHeight, imageWidth = get_image_size(image)
    numberOfElementsToChange = math.ceil(augmentationFactor * (imageHeight * imageWidth))

    # Kill pixels n number of times
    for n in range(numberOfElementsToChange):
        x, y = pick_alteration_coordinates(imageHeight, imageWidth)
        # Black out randomly selected pixels
        image[x, y] = [0, 0, 0]
    return image


# Determine whether given file is an image or not
def file_is_image(path_to_file):
    filename, extension = os.path.splitext(path_to_file)
    if extension != '.jpg':
        return False
    else:
        return True


def get_image_size(image):
    height, width, channels = image.shape
    return height, width

# Control Variables
# imageHeight = 56
# imageWidth = 56
augmentationFactor = 0.01  # Decimal percentage
newImagePath = "datasets/cats-dogs-noise-001/dog"
datasetDirectory = "datasets/cats-dogs/dog"

# Go through the files in a dataset sub-directory
for file in os.listdir(datasetDirectory):

    if file_is_image(datasetDirectory + "/" + file):
        print(file)
        # Load the original image
        originalImage = cv2.imread(datasetDirectory + '/' + file)
        # Change the image
        newImage = alter_image(originalImage, augmentationFactor)
        # Save the image in new directory with the same name
        newFilename = newImagePath + '/' + file
        save_image(newImage, newFilename)

print("Augmentation Complete")
