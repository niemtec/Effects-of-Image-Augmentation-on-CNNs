import datetime
import matplotlib.pyplot as plt
import os

graph_size = (15, 10)


# Prints current timestamp, to be used in print statements
def stamp():
	time = "[" + str(datetime.datetime.now().time()) + "]   "
	return time


# Save final model performance
def saveNetworkStats(history, epochs, initial_learning_rate, model_name, results_path):
	# Extract data from history dictionary
	history_loss = history.history['loss']
	history_loss = str(history_loss[-1])  # Get last value from loss
	history_acc = history.history['acc']
	history_acc = str(history_acc[-1])  # Get last value from accuracy
	history_val_loss = history.history['val_loss']
	history_val_loss = str(history_val_loss[-1])
	history_val_acc = history.history['val_acc']
	history_val_acc = str(history_val_acc[-1])
	history_mse = history.history['mean_squared_error']
	history_mse = str(history_mse[-1])
	history_lr = history.history['lr']
	history_lr = str(history_lr[-1])
	history_val_lr = history.history['val_lr']
	history_val_lr = str(history_val_lr[-1])
	
	with open(results_path + model_name + ".txt", "a") as history_log:
		history_log.write(
			model_name + "," + history_loss + "," + history_acc + "," + history_val_loss + "," + history_val_acc + "," + str(
				epochs) + "," + str(initial_learning_rate) + "," + str(history_mse) + "," + history_lr + "," + history_val_lr + "\n")
	history_log.close()
	
	print(stamp() + "Keras Log Saved")
	print(history.history.keys())
	print(stamp() + "History File Saved")


def saveFigureStats(fig, figure_name, model_name, results_path):
	fig.savefig(results_path + '/' + model_name + '-' + figure_name + '.png')


# Summarize history for accuracy
def saveAccuracyGraph(history, model_name, results_path):
	plt.figure(figsize = graph_size, dpi = 75)
	plt.grid(True, which = 'both')
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	plt.suptitle(model_name)
	saveFigureStats(plt, 'accuracy', model_name, results_path)
	plt.close()


# Summarize history for loss
def saveLossGraph(history, model_name, results_path):
	plt.figure(figsize = graph_size, dpi = 75)
	plt.grid(True, which = 'both')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	plt.suptitle(model_name)
	saveFigureStats(plt, 'loss', model_name, results_path)
	plt.close()
	
def saveLearningRateGraph(history, model_name, results_path):
	plt.figure(figsize = graph_size, dpi = 75)
	plt.grid(True, which = 'both')
	plt.plot(history.history['lr'])
	plt.plot(history.history['val_lr'])
	plt.title('Learning Rate')
	plt.ylabel('learning_rate')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')
	plt.suptitle(model_name)
	saveFigureStats(plt, 'learning_rate', model_name, results_path)
	plt.close()


def saveModelToDisk(model, model_name, results_path):
	print(stamp() + "Saving Network Model")
	model_json = model.to_json()
	with open(results_path + '/' + model_name + ".json", "w") as json_file:
		json_file.write(model_json)


def saveWeightsToDisk(model, model_name, results_path):
	print(stamp() + "Saving Network Weights")
	model.save_weights(results_path + '/' + model_name + ".h5", "w")


@staticmethod
def isFileAnImage(path_to_file):
	filename, extension = os.path.splitext(path_to_file)
	if extension != '.jpg':
		return False
	else:
		return True
