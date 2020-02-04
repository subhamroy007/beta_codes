#import all the nessessery libraries

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt


#define the function to calculate loss function
def log_loss_val(y,pred):
    return -y[0][0]*np.log10(pred[0][0]) - (1-y[0][0])*np.log10(1-pred[0][0])

#define the activation sigmoid function
def sigmoid(x):
    return (1+np.exp(-x)) ** -1

#define weight initializer
def init_weight(size):
    w = []
    for x in range(size):
        w.append(np.random.randn())
    return np.array(w,dtype = np.float64).reshape(1,size)



class perceptron:
    def __init__(self,input_size,learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = init_weight(self.input_size)
        self.bias = np.array(np.random.randn(),dtype = np.float64).reshape(1,1)
        self.loss_list = []
    
    def train(self,data_x,data_y):
        for x,y in zip(data_x,data_y):
            sum = np.dot(self.weights,x) + self.bias
            acv = sigmoid(sum)
            loss = log_loss_val(y,acv)
            self.loss_list.append(loss)
            print("loss = ",loss)
            self.weights = self.weights-self.learning_rate*np.dot((acv-y),x.T)
            self.bias = self.bias - self.learning_rate*(acv-y)
            
      
    def sketch(self,data_x,data_y):
        print("this is the original distribution")
        for x,y in zip(data_x,data_y):
            if y[0][0] == 1.0:
                plt.scatter(x[0][0],x[1][0],color = "green")
            else:
                plt.scatter(x[0][0],x[1][0],color = "yellow")
        plt.show()
        
        print("this is the predicted distribution")
        for x,y in zip(data_x,data_y):
            if round(sigmoid(np.dot(self.weights,x) + self.bias)[0][0]) == 1.0:
                plt.scatter(x[0][0],x[1][0],color = "green")
            else:
                plt.scatter(x[0][0],x[1][0],color = "yellow")
        plt.show()
        
    def accuracy(self,data_x,data_y):
        right = 0
        wrong = 0
        for x,y in zip(data_x,data_y):
            if round(sigmoid(np.dot(self.weights,x) + self.bias)[0][0]) == y[0][0]:
                right += 1
            else:
                wrong += 1
                
        print("accuracy = ",right/(right+wrong))

data = pd.read_csv('log_data.csv')
data_x = np.array(data.iloc[:,[2,3]].values,dtype = np.float64).reshape(400,2,1)
data_y = np.array(data.iloc[:,[4]].values,dtype = np.float64).reshape(400,1,1)
mean = np.mean(data_x,axis = 0)
sd = np.var(data_x,axis = 0)**.5
data_x[:,0] = (data_x[:,0]-mean[0])/sd[0]
data_x[:,1] = (data_x[:,1]-mean[1])/sd[1]
my_model = perceptron(2,.042)

my_model.train(data_x,data_y)
#print("weights = {},{},{}".format(my_model.w0,my_model.w1,my_model.w2))
my_model.sketch(data_x,data_y)
my_model.accuracy(data_x,data_y)
plt.plot(my_model.loss_list)
plt.show()


