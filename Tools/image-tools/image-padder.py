# Script used to add padding to dataset images for input standarisation
import os

from PIL import Image, ImageOps


def padImage(path_to_image, path_to_processed_directory):
    desired_size = 1024
    im_path = path_to_image

    im = Image.open(im_path)
    old_size = im.size # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # Create a  new i mage and paste the resized onto it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

    #new_im.show()
    path, new_filename = os.path.split(path_to_image)

    new_im.save((path_to_processed_directory + "//" + new_filename), "PNG")

working_directory = "C://Users//janie//PycharmProjects//Project-Turing//validation//malignant"

for file in os.listdir(working_directory):
    print(file)
    output_path = "C://Users//janie//PycharmProjects//Project-Turing//padded-dataset//validation//malignant"
    padImage(working_directory + "//" + file, output_path)

