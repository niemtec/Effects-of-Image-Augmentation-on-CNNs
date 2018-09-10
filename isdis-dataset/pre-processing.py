from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Image distortion preprocessing
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

img = load_img('/home/niemtec/tmp/Project-Turing/isdis-dataset/benign/ISIC_0000003.jpg')
x = img_to_array(img) # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='/home/niemtec/tmp/Project-Turing/isdis-dataset/preview', save_prefix='preview', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely