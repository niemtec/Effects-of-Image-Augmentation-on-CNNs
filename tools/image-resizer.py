# Script used to add padding to dataset images for input standarisation
import os

from PIL import Image, ImageOps

def resizeImage(path_to_image, path_to_processed_directory):
    desired_size = (64, 64)

    original_image = Image.open(path_to_image)
    new_image = ImageOps.fit(original_image, desired_size, Image.ANTIALIAS)
    path, new_filename = os.path.split(path_to_image)

    new_image.save((path_to_processed_directory + "//" + new_filename), "PNG")

working_directory = "datasets/cats-dogs/cat"

for file in os.listdir(working_directory):
    print(file)
    output_path = "datasets/cats-dogs-resized/cat"
    if(file != "desktop.ini"):
        resizeImage(working_directory + "//" + file, output_path)

