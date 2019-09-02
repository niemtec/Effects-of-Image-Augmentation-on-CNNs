import datetime
import matplotlib.pyplot as plt
import os


class Helper(object):
	graph_size = (15, 10)

	def __init__(self, results_path, model_name):
		self.results_path = results_path
		self.model_name = model_name

	# Prints current timestamp, to be used in print statements
	def stamp(self):
		time = "[" + str(datetime.datetime.now().time()) + "]   "
		return time

	# Save final model performance
	def saveNetworkStats(self, history, epochs, initial_learning_rate):
		# Extract data from history dictionary
		history_loss = history.history['loss']
		history_loss = str(history_loss[-1])  # Get last value from loss
		history_acc = history.history['acc']
		history_acc = str(history_acc[-1])  # Get last value from accuracy
		history_val_loss = history.history['val_loss']
		history_val_loss = str(history_val_loss[-1])
		history_val_acc = history.history['val_acc']
		history_val_acc = str(history_val_acc[-1])
		history_mse = str(history.history['mse'])

		with open(self.results_path + self.model_name + ".txt", "a") as history_log:
			history_log.write(
				self.model_name + "," + history_loss + "," + history_acc + "," + history_val_loss + "," + history_val_acc + "," + str(epochs) + "," + str(initial_learning_rate) + "," + str(history_mse) + "\n")
		history_log.close()

		print(self.stamp() + "Keras Log Saved")
		print(history.history.keys())
		print(self.stamp() + "History File Saved")

	def saveFigureStats(self, fig, figure_name):
		fig.savefig(self.results_path + '/' + self.model_name + '-' + figure_name + '.png')

	# Summarize history for accuracy
	def saveAccuracyGraph(self, history):
		plt.figure(figsize = self.graph_size, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(self.model_name)
		self.saveFigureStats(plt, 'accuracy')
		plt.close()

	# Summarize history for loss
	def saveLossGraph(self, history):
		plt.figure(figsize = self.graph_size, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(self.model_name)
		self.saveFigureStats(plt, 'loss')
		plt.close()
	
	def saveModelToDisk(self, model):
		print(self.stamp() + "Saving Network Model")
		model_json = model.to_json()
		with open(self.results_path + '/' + self.model_name + ".json", "w") as json_file:
			json_file.write(model_json)
		
	def saveWeightsToDisk(self, model):
		print(self.stamp() + "Saving Network Weights")
		model.save_weights(self.results_path + '/' + self.model_name + ".h5", "w")
	
	@staticmethod
	def isFileAnImage(path_to_file):
		filename, extension = os.path.splitext(path_to_file)
		if extension != '.jpg':
			return False
		else:
			return True
