import matplotlib.pyplot as plt
import numpy as np

class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# Compute the set of learning rates for each corresponding epoch
		lrs =[self(i) for i in epochs]
		
		# Learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")