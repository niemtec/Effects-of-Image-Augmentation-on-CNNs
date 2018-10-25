# Script used to add padding to dataset images for input standarisation
import os

from PIL import Image, ImageOps

def resizeImage(path_to_image, path_to_processed_directory):
    desired_size = 512, 512

    im = Image.open(path_to_image)
    im.thumbnail(desired_size, Image.ANTIALIAS)
    path, new_filename = os.path.split(path_to_image)

    im.save((path_to_processed_directory + "//" + new_filename), "PNG")

working_directory = "padded-dataset//validation//benign"

for file in os.listdir(working_directory):
    print(file)
    output_path = "rescaled-dataset-512//validation//benign"
    resizeImage(working_directory + "//" + file, output_path)

