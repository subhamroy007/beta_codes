import numpy as np 
import pandas
import random
import math
import matplotlib.pyplot as plt 

class perceptron:
	def __init__(self,input_size,output_size,learning_rate):
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = learning_rate
		self.weights = self.init_weights()
		self.bias = np.random.randn()
		print(self.weights)
		print(self.bias)


	def init_weights(self):
		w = []
		for x in range(self.input_size):
			w.append(np.random.randn())
		return w

	def train(self,data_x,data_y):
		for i in range(data_x.shape[0]):
			x = data_x[i].tolist()
			y = data_x[i].tolist()
			x1 = 0.0
			for j in range(self.input_size):
				x1 += self.weights[j]*x[j]
			x1 += self.bias
			x2 = (1+np.exp(-x1)) ** -1
			x3 = (y[0]-x2) ** 2
			print("loss = ",x3)
			x4 = x2*(1-x2)
			x5 = 2*(y[0]-x2)
			self.bias -= self.learning_rate * (x4*x5)
			for j in range(self.input_size):
				self.weights[j] -= self.learning_rate*(x4*x5*x[j])


	@classmethod

	def standardization(self,x):
		mean = np.mean(x,axis = 0)
		sd = np.var(x,axis = 0) ** .5
		for i in range(x.shape[1]):
			x[:,i] = x[:,i] - mean[i]
			x[:,i] = x[:,i] / sd[i]
		return x


data = pandas.read_csv("log_data.csv")
x = np.array(data.iloc[:, [2, 3]].values,dtype = np.float64)
y = data.iloc[:, [4]].values
x = perceptron.standardization(x)
new_p = perceptron(2,1,0.001)
new_p.train(x,y)