from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range = 30,
    fill_mode = 'nearest'
)

img = load_img('datasets/cats-dogs-resized/cats-dogs-noise-000/cat/cat.277.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir = 'datasets/image-corruption-dataset/cats-dogs-rotation-samples/',
                          save_prefix = 'cat-30',
                          save_format = 'jpeg'):
    i += 1
    if i > 0:
        break
