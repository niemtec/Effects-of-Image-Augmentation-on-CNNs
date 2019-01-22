# Tool used to print the dataset images for pictographic representation of data augmentation techniques
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2

# Control Variables
imageHeight = 28
imageWidth = 28


def load_image(pathToImage):
    # Load the image
    image = cv2.imread(pathToImage)
    # Resize the image
    image = cv2.resize(image, imageHeight, imageWidth)
    # Convert image to array
    imageArray = img_to_array(image)
    return imageArray


def plot_image(imageArray, saveDirectory, newImageName):
    # DEBUG: Show image if you want
    plt.imshow(imageArray / 255)  # /255 to make image coloured again
    plt.savefit(saveDirectory + "/" + newImageName + ".png")

########################################################
# Load image to display how array looks like
