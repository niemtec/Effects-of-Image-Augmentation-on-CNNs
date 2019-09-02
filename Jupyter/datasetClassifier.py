
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
from classifierHelpers import Helper

Tools = Helper(None, None)

dataset_path = '../Cancer-Dataset/'

# Initialize the data and labels arrays
sorted_data = []
sorted_labels = []
data = []
labels = []

# Go through dataset directory
print(Tools.stamp() + "Classifying the Dataset")
for dataset_category in os.listdir(dataset_path):
	dataset_category_path = dataset_path + "/" + dataset_category
	
	if not dataset_category.startswith('.'):
		for sample in os.listdir(dataset_category_path):
			print(Tools.stamp() + sample)
			
			if Tools.isFileAnImage(dataset_category_path + "/" + sample):
				image = cv2.imread(dataset_category_path + "/" + sample)
				image = cv2.resize(image, (64, 64))
				image = img_to_array(image)
				sorted_data.append(image)
				
				if dataset_category == 'benign':
					sorted_labels.append(1)
				else:
					sorted_labels.append(0)
					
np.save('sorted_data_array', sorted_data)
np.save('sorted_labels_array', sorted_labels)
print('Saving complete')