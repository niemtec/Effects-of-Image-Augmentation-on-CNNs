# Import Libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np

# Process Model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.h5")
print("Loaded model from disk")

model = loaded_model
image = load_img('cat.jpg', target_size = (64, 64))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

# Generate predictions
pred = model.predict(image)
# print('Predicted:', decode_predictions(pred, top=2)[0])
np.argmax(pred[0])

# Grad-CAM algorithm
specoutput = model.output[:, 1]
last_conv_layer = model.get_layer('max_pooling2d_2')
grads = K.gradients(specoutput, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis = (0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([image])
for i in range(50):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis = -1)

# Heatmap post processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
