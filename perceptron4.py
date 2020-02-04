#import all the nessessery libraries

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt


#define the function to calculate loss function
def log_loss_val(y,pred):
    return -y*np.log10(pred) - (1-y)*np.log10(1-pred)

#define the activation sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))



#definnig the perceptron class
class perceptron:
    def __init__(self):
        self.input_size = 2
        self.learning_rate = .04
        self.output_size = 1
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w0 = np.random.randn()
        self.loss_list = []
        
    def train(self,data_x,data_y):
        for x,y in zip(data_x,data_y):
            #print("training dataset")
            sum = x[0]*self.w0 + x[1]*self.w1 + self.w2
            acv = sigmoid(sum)
            loss = log_loss_val(y,acv)
            self.loss_list.append(loss)
            print("y = {} and predicted = {} and loss = {}".format(y,acv,loss))
            self.w0 -= self.learning_rate*x[0]*(acv-y)
            self.w1 -= self.learning_rate*x[1]*(acv-y)
            self.w2 -= self.learning_rate*(acv-y)
            
    def sketch(self,data_x,data_y):
        print("this is the original distribution")
        for x,y in zip(data_x,data_y):
            if y == 1.0:
                plt.scatter(x[0],x[1],color = "green")
            else:
                plt.scatter(x[0],x[1],color = "yellow")
        plt.show()
        
        print("this is the predicted distribution")
        for x,y in zip(data_x,data_y):
            if round(sigmoid(x[0]*self.w0 + x[1]*self.w1 + self.w2)) == 1.0:
                plt.scatter(x[0],x[1],color = "green")
            else:
                plt.scatter(x[0],x[1],color = "yellow")
        plt.show()
        
    def accuracy(self,data_x,data_y):
        right = 0
        wrong = 0
        for x,y in zip(data_x,data_y):
            if round(sigmoid(x[0]*self.w0 + x[1]*self.w1 + self.w2)) == y:
                right += 1
            else:
                wrong += 1
                
        print("accuracy = ",right/(right+wrong))


data = pd.read_csv('log_data.csv')
data_x = np.array(data.iloc[:,[2,3]].values,dtype = np.float64)
data_y = np.array(data.iloc[:,4].values,dtype = np.float64)
mean = np.mean(data_x,axis = 0)
sd = np.var(data_x,axis = 0)**.5
data_x[:,0] = (data_x[:,0]-mean[0])/sd[0]
data_x[:,1] = (data_x[:,1]-mean[1])/sd[1]
my_model = perceptron()
#my_model.sketch(data_x,data_y)
#my_model.accuracy(data_x,data_y)
#print("weights = {},{},{}".format(my_model.w0,my_model.w1,my_model.w2))
my_model.train(data_x,data_y)
#print("weights = {},{},{}".format(my_model.w0,my_model.w1,my_model.w2))
my_model.sketch(data_x,data_y)
my_model.accuracy(data_x,data_y)
plt.plot(my_model.loss_list)
plt.show()