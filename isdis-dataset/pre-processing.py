import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Image distortion preprocessing
datagen = ImageDataGenerator(
    rotation_range=40,
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True)

directory = "C://Users//janie//Documents//GitHub//Project-Turing//isdis-dataset//train//benign"
for image in os.listdir(directory):
    print("Processing Image: " + image)
    img = load_img(directory + "//" + image)
    x = img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='C://Users//janie//Documents//GitHub//Project-Turing//isdis-dataset//train//benign_aug', save_prefix='aug_', save_format='jpeg'):
        i += 1
        print("    >  Augmenting Batch: " + str(i))
        if i > 4:
            break  # otherwise the generator would loop indefinitely