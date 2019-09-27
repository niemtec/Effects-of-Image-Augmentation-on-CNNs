from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range = 180,
    fill_mode = 'nearest'
)

img = load_img('tools/network-heatmap-visualisation/malignant - rotation/0.jpg')

x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir = 'tools/network-heatmap-visualisation/malignant - rotation/',
                          save_prefix = 'benign',
                          save_format = 'jpeg'):
    i += 1
    if i > 100:
        break