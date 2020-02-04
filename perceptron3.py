import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

data = pd.read_csv("log_data.csv")

x = data.iloc[:,[2,3]].values
y = data.iloc[:,[4]].values

class perceptron:
    def __init__(self,input_size,output_size,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = self.init_weights()
        self.bias = np.random.randn()
        self.losses = []
    
    def init_weights(self):
        w = []
        for x in range(self.input_size):
            w.append(np.random.randn())
        return w
    
    def train(self,data_x,data_y):
        mean = np.mean(data_x,axis = 0)
        sd = np.var(data_x,axis = 0) ** .5
        for i in range(data_x.shape[1]):
            data_x[:,i] = data_x[:,i]-mean[i]
            data_x[:,i] = data_x[:,i]/sd[i]
        for i in range(data_x.shape[0]):
            sum = 0
            for j in range(self.input_size):
                sum += self.weights[j]*data_x[:,j][i]
            sum += self.bias
            acv = (1+np.exp(-sum)) ** -1
            loss = (data_y[:,0][i]-acv) ** 2
            print("training no ",i+1," loss = ",loss)
            self.losses.append(loss)
            for j in range(self.input_size):
                self.weights[j] += self.learning_rate*data_x[:,j][i]*acv*(1-acv)*2*(data_y[:,0][i]-acv)
                self.bias += self.learning_rate*acv*(1-acv)*2*(data_y[:,0][i]-acv)
                
                
new_obj = perceptron(2,1,.14)
new_obj.train(x,y)



plt.plot(new_obj.losses)



