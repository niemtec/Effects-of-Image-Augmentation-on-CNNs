import datetime
import matplotlib.pyplot as plt
import os


class Helper(object):
	graphSize = (15, 10)

	def __init__(self, resultsPath, modelName):
		self.resultsPath = resultsPath
		self.modelName = modelName

	# Prints current timestamp, to be used in print statements
	def stamp(self):
		time = "[" + str(datetime.datetime.now().time()) + "]   "
		return time

	# Save final model performance
	def saveNetworkStats(self, history, noEpochs, initialLearningRate):
		# Extract data from history dictionary
		historyLoss = history.history['loss']
		historyLoss = str(historyLoss[-1])  # Get last value from loss
		historyAcc = history.history['acc']
		historyAcc = str(historyAcc[-1])  # Get last value from accuracy
		historyValLoss = history.history['val_loss']
		historyValLoss = str(historyValLoss[-1])
		historyValAcc = history.history['val_acc']
		historyValAcc = str(historyValAcc[-1])
		historyMSE = 0  # str(historyMSE[-1])

		with open(self.resultsPath + self.modelName + ".txt", "a") as history_log:
			history_log.write(
				self.modelName + "," + historyLoss + "," + historyAcc + "," + historyValLoss + "," + historyValAcc + "," + str(noEpochs) + "," + str(initialLearningRate) + "," + str(historyMSE) + "\n")
		history_log.close()

		print(self.stamp() + "Keras Log Saved")
		print(history.history.keys())
		print(self.stamp() + "History File Saved")

	def saveFigureStats(self, fig, figureName):
		fig.savefig(self.resultsPath + '/' + self.modelName + '-' + figureName + '.png')

	# Summarize history for accuracy
	def saveAccuracyGraph(self, history):
		plt.figure(figsize = self.graphSize, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(self.modelName)
		self.saveFigureStats(plt, 'accuracy')
		plt.close()

	# Summarize history for loss
	def saveLossGraph(self, history):
		plt.figure(figsize = self.graphSize, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(self.modelName)
		self.saveFigureStats(plt, 'loss')
		plt.close()
	
	def saveModelToDisk(self, model):
		print(self.stamp() + "Saving Network Model")
		model_json = model.to_json()
		with open(self.resultsPath + '/' + self.modelName + ".json", "w") as json_file:
			json_file.write(model_json)
		
	def saveWeightsToDisk(self, model):
		print(self.stamp() + "Saving Network Weights")
		model.save_weights(self.resultsPath + '/' + self.modelName + ".h5", "w")
	
	@staticmethod
	def isFileAnImage(path_to_file):
		filename, extension = os.path.splitext(path_to_file)
		if extension != '.jpg':
			return False
		else:
			return True
