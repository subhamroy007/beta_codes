import numpy as np 
import pandas
import random
import math
import matplotlib.pyplot as plt 

#creating a perceptron class
class perceptron:
	def __init__(self,input_size,output_size,learning_rate):
		self.input_size = input_size
		self.output_size = output_size
		self.learning_rate = learning_rate
		self.weight = self.init_weight(self.input_size)
		self.bias = self.init_bias()
		self.loss_list = []
		self.dw = None
		self.db = None

	def forward_pass(self,x,y):
		x = np.array(x,dtype = np.float64).reshape((self.input_size,1))
		y = np.array(y,dtype = np.float64).reshape((self.output_size,1))
		self.dw = x
		self.db = None
		x = self.linear_cal(self.weight,self.bias,x)
		x1 = self.sigmoid_cal(x)
		x = self.loss_cal(x1,y)
		print("new loss = ",x)
		self.loss_list.append(x)
		x2 = 2 * (y-x1)
		self.dw = np.dot(self.dw,(x1*x2))
		self.db = x1*x2



	def backward_pass(self,x,y):
		self.weight -= self.dw * self.learning_rate
		self.bias -= self.db * self.learning_rate

	def eval(self,data_x,data_y):
		for x,y in zip(data_x,data_y):
			self.forward_pass(x,y)
			self.backward_pass(x,y)

		
	@classmethod

	def standardization(self,x):
		mean = np.mean(x,axis = 0)
		sd = np.var(x,axis = 0) ** .5
		for i in range(x.shape[1]):
			x[:i] = x[:i] - mean[i]
			x[:i] = x[:i] / sd[i]
		return x

	@classmethod 

	def linear_cal(self,weight,bias,x):
		return np.dot(weight.T,x)+bias

	@classmethod

	def sigmoid_cal(self,x):
		return (1+np.exp(-x)) ** -1

	@classmethod
	
	def loss_cal(self,prob,y):
		return (y-prob) ** 2

	@classmethod

	def init_weight(self,size):
		w = []
		for x in range(size):
			w.append(np.random.randn())
		return np.array(w).reshape((size,1))


	@classmethod

	def init_bias(self):
		return np.array([np.random.randn()]).reshape((1,1))




data = pandas.read_csv("log_data.csv")
x = np.array(data.iloc[:, [2, 3]].values,dtype = np.float64)
y = data.iloc[:, [4]].values
x = perceptron.standardization(x)
new_p = perceptron(2,1,0.01)
new_p.eval(x,y)
