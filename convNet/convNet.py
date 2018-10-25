# An attempt at using Transfer Learning to approach the Cancer Identification Problem

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width = 256
img_height = 256
train_data_dir = 'padded-dataset/train'
validation_data_dir = 'padded-dataset/validation'
nb_train_samples = 512
nb_validation_samples = 512
batch_size = 16
epochs = 50

model = applications.VGG19(
   weights = 'imagenet',
   include_top = False,
   input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train
for layer in model.layers[:5]:
   layer.trainable = False

# Adding custom layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(16, activation = 'softmax')(x)

# Creating the final model
model_final = Model(
   input = model.input,
   output = predictions)

# Compile the model
model_final.compile(
   loss = 'binary_crossentropy',
   optimizer = optimizers.SGD(lr = 0.0001, momentum = 0.9),
   metrics = ['accuracy'])

# Initiate the train and test generators with data Augmentation
train_datagen = ImageDataGenerator(
   rescale = 1. / 255,
   horizontal_flip = True,
   vertical_flip = True,
   fill_mode = 'nearest',
   zoom_range = 0.3,
   rotation_range = 30)

validation_datagen = ImageDataGenerator(
   rescale = 1. / 255,
   horizontal_flip = True,
   vertical_flip = True,
   fill_mode = 'nearest',
   zoom_range = 0.3,
   rotation_range = 30)

train_generator = train_datagen.flow_from_directory(
   train_data_dir,
   target_size = (img_height, img_width),
   batch_size = batch_size,
   class_mode = 'binary')

validation_generator = validation_datagen.flow_from_directory(
   validation_data_dir,
   target_size = (img_height, img_width),
   batch_size = batch_size,
   class_mode = 'binary')

# Save the model according to the conditions
checkpoint = ModelCheckpoint("convNet/vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='convNet/val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

