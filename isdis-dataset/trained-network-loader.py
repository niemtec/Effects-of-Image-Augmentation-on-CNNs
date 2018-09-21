from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# Load and create the model
json_file = open("C://Users//janie//PycharmProjects//Project-Turing//isdis-dataset//classifier-network-v1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights("C://Users//janie//PycharmProjects//Project-Turing//isdis-dataset//85-95.h5")
print("Loaded Model from Disk")

# Evaluate loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score  = loaded_model.evaluate()