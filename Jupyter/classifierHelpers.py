import datetime
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np


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
	def save_network_stats(self, resultsPath, modelName, history, fileName, sensitivity, specificity, precision, noEpochs, initialLearningRate):
		# Extract data from history dictionary
		historyLoss = history.history['loss']
		historyLoss = str(historyLoss[-1])  # Get last value from loss
		historyAcc = history.history['acc']
		historyAcc = str(historyAcc[-1])  # Get last value from accuracy
		historyValLoss = history.history['val_loss']
		# Get last value from validated loss
		historyValLoss = str(historyValLoss[-1])
		historyValAcc = history.history['val_acc']
		# Get last value from validated accuracy
		historyValAcc = str(historyValAcc[-1])
		historyMSE = 0  # str(historyMSE[-1])
		historyMAPE = 0  # history.history['mape']
		historyMAPE = 0  # str(historyMAPE[-1])

		with open(resultsPath + fileName + ".txt", "a") as history_log:
			history_log.write(
				modelName + "," + historyLoss + "," + historyAcc + "," + historyValLoss + "," + historyValAcc + "," + str(
					noEpochs) + "," + str(initialLearningRate) + "," + str(historyMSE) + "," + str(
					historyMAPE) + "," + str(sensitivity) + "," + str(specificity) + "," + str(precision) + "\n")
		history_log.close()

		print(self.stamp() + "Keras Log Saved")

		print(history.history.keys())

		print(self.stamp() + "History File Saved")

	# Calculate confusion matrix statistics
	@staticmethod
	def calculate_statistics(tn, fp, fn, tp):
		sensitivity = tp / (tp + fn)
		specificity = tn / (fp + tn)
		precision = tp / (tp + fp)

		return sensitivity, specificity, precision

	def save_figure(self, fig, figureName):
		fig.savefig(self.resultsPath + '/' + self.modelName + '-' + figureName + '.png')

	# Save the confusion matrix as a graphical figure
	def save_confusion_matrix(self, tp, tn, fp, fn, resultsPath, modelName):
		import seaborn as sns
		tp = int(tp)
		tn = int(tn)
		fp = int(fp)
		fn = int(fn)

		cm = [[tp, tn], [fp, fn]]
		cm = np.array(cm)
		heatmap = sns.heatmap(cm, annot = True, fmt = 'g', linewidths = 0.2)
		fig = heatmap.get_figure()
		self.save_figure(fig, 'confusion-matrix')

	# Summarize history for accuracy
	def save_accuracy_graph(self, history, modelName):
		plt.figure(figsize = self.graphSize, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(modelName)
		self.save_figure(plt, 'accuracy')
		plt.close()

	# Summarize history for loss
	def save_loss_graph(self, history, modelName):
		plt.figure(figsize = self.graphSize, dpi = 75)
		plt.grid(True, which = 'both')
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc = 'upper left')
		plt.suptitle(modelName)
		self.save_figure(plt, 'loss')
		plt.close()

	@staticmethod
	def file_is_image(path_to_file):
		filename, extension = os.path.splitext(path_to_file)
		if extension != '.jpg':
			return False
		else:
			return True
