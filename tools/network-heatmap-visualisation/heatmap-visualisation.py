from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import cv2

imagePath = 'malignant - rotation/45.jpeg'
modelName = 'trained-networks/all-corrupted-45.json'
weightsName = 'trained-networks/all-corrupted-45.h5'
heatmapDirectory = 'malignant - rotation/'

jsonFile = open(modelName, 'r')
loadedModelJSON = jsonFile.read()
jsonFile.close()
loadedModel = model_from_json(loadedModelJSON)
loadedModel.load_weights(weightsName)

model = loadedModel
img = image.load_img(imagePath, target_size = (64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

preds = model.predict(x)
classIDX = np.argmax(preds[0])
classOutput = model.output[:, classIDX]
lastLayer = model.get_layer("max_pooling2d_2")

grads = K.gradients(classOutput, lastLayer.output)[0]
pooledGrads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooledGrads, lastLayer.output[0]])
pooledGradsValue, convolutionalLayerOutputValue = iterate([x])
for i in range(49):
    convolutionalLayerOutputValue[:, :, i] *= pooledGradsValue[i]

heatmap = np.mean(convolutionalLayerOutputValue, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = cv2.imread(imagePath)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposedImage = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
# cv2.imshow("Original", img)
# cv2.imshow("GradCam", superimposedImage)

cv2.imwrite(imagePath + "-heatmap.jpg", superimposedImage)
