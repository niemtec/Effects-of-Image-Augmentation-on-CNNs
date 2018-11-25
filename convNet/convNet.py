# An attempt at using Transfer Learning to approach the Cancer Identification Problem

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

img_width = 150
img_height = 150
train_data_dir = '../datasets/cats-dogs'
validation_data_dir = '../datasets/cats-dogs'
nb_train_samples = 1000
nb_train_samples = 1000
nb_train_samples = 1000
nb_validation_samples = 1000
batch_size = 30
epochs = 200
model_name = 'convnet-200-epochs'

model = applications.VGG19(
   weights = 'imagenet',
   include_top = False,
   input_shape = (img_width, img_height, 3))

# Configuration Parameters for Freezing Layers
################################################################################################################
# Freeze the layers which you don't want to train
# for layer in model.layers[:5]:
#    layer.trainable = False

################################################################################################################

# Adding custom layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(1, activation = 'softmax')(x)

# Creating the final model
model_final = Model(
   input = model.input,
   output = predictions)

# Compile the model
model_final.compile(
   loss = 'binary_crossentropy',
   optimizer = optimizers.SGD(lr = 0.0001, momentum = 0.9),
   metrics = ['accuracy'])

# Initiate the train and validation generators with data Augmentation
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
checkpoint = ModelCheckpoint("../convNet/vgg16_1_raw_dataset.h5", monitor = 'val_acc', verbose = 1, save_best_only = True,
                             save_weights_only = False, mode = 'auto', period = 1)
early = EarlyStopping(monitor = '../convNet/val_acc', min_delta = 0, patience = 10, verbose = 1, mode = 'auto')

# Train the model
history = model_final.fit_generator(
   train_generator,
   samples_per_epoch = nb_train_samples,
   epochs = epochs,
   validation_data = validation_generator,
   nb_val_samples = nb_validation_samples,
   callbacks = [checkpoint, early])

print(history.history.keys())
# Summarize history for accuracy
plt.figure(figsize = graph_size, dpi = 300)
plt.grid(True, which = 'both')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.savefig('Results/' + model_name + "-accuracy.png")
plt.close()

# Summarize history for loss
plt.figure(figsize = graph_size, dpi = 300)
plt.grid(True, which = 'both')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.savefig('Results/' + model_name + "-loss.png")
plt.close()
