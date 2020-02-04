#creating a 1-hidden layer neural network with same activation function,
#optimizer and loss to see the change in the output


#import necessery libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


#creating the uniform computation node
class Uniform:
    def __init__(self,W,B):
        self.w = W
        self.x = None
        self.b = B
        self.forward_val = None
        self.db = None
        self.dx = None
        self.dw = None
        
    def forwardVal(self,x):
        self.x = x
        self.forward_val = np.dot(self.w,self.x) + self.b
        return self.forward_val

    def backwardVal(self):
        self.dw = self.x
        self.dx = self.w
        self.db = 1 
        

#creating the relu computation node
class Relu:
    def __init__(self):
        self.x = None
        self.forward_val = None
        self.dx = None
    def forwardVal(self,x):
        self.x = x
        self.forward_val = self.x
        for i in range(self.forward_val.shape[0]):
            for j in range(self.forward_val.shape[1]):
                self.forward_val[i,j] = max(0,self.forward_val[i,j])
        return self.forward_val
    def backwardVal(self):
        self.dx = self.forward_val
        for i in range(self.forward_val.shape[0]):
            for j in range(self.forward_val.shape[1]):
                if self.forward_val[i,j] == 0:
                    self.dx[i,j] = 0
                else:
                    self.dx[i,j] = 1
        
                    
#creating the sigmoid node
class Sigmoid:
    def __init__(self):
        self.x = None
        self.dx = None
        self.forward_val = None
    
    def forwardVal(self,x):
        self.x = x
        self.forward_val = (1 + np.exp(-self.x)) ** -1
        return self.forward_val
    
    def backwardVal(self):
        self.dx = self.forward_val * (1-self.forward_val)
        


#creating the loss node
class LogLoss:
    def __init__(self):
        self.y = None
        self.pred = None
        self.l = None
        self.dl = None
        
    def calculateLoss(self,y,pred):
        self.y = y
        self.pred = pred
        self.l = self.y*np.log10(self.pred)+(1-self.y)*np.log10(1-self.pred)
        self.l = -self.l
        return self.l
    
    def calculateGradient(self):
        self.dl = (self.y-self.pred) / (self.pred * (1-self.pred))



#creating the neural network architecture
class Network:
    def __init__(self,input_size,layer_size,output_size,learning_rate):
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layer_stack = []
        self.loss_list = []
        
        
        #initializing all the neccessery computation nodes 
        self.layer_stack.append(Uniform(self.init_weight(self.layer_size,self.input_size),self.init_bias(self.layer_size,1)))
        self.layer_stack.append(Relu())
        self.layer_stack.append(Uniform(self.init_weight(self.output_size,self.layer_size),self.init_bias(self.output_size,1)))
        self.layer_stack.append(Sigmoid())
        self.layer_stack.append(LogLoss())

    @classmethod
    def init_weight(self,r,c):
        arr = np.zeros((r,c),dtype = np.float64)
        for i in range(r):
            for j in range(c):
                arr[i,j] = np.random.randn() #initialize by gausian 
        return arr
   
    @classmethod
    def init_bias(self,r,c):
        arr = np.zeros((r,c),dtype = np.float64)
        for i in range(r):
            for j in range(c):
                arr[i,j] = np.random.randn() #initialize by gausian 
        return arr
   

    def forwardPass(self,sample_x,sample_y):
        temp = self.layer_stack[0].forwardVal(sample_x)
        temp = self.layer_stack[1].forwardVal(temp)
        temp = self.layer_stack[2].forwardVal(temp)
        temp = self.layer_stack[3].forwardVal(temp)
        temp = self.layer_stack[4].calculateLoss(sample_y,temp)
        self.loss_list.append(temp)
        print("loss = ",temp)
        
    def backwardPass(self):
        self.layer_stack[0].backwardVal()
        self.layer_stack[1].backwardVal()
        self.layer_stack[2].backwardVal()
        self.layer_stack[3].backwardVal()
        self.layer_stack[4].calculateGradient()
        
        self.layer_stack[2].w = self.layer_stack[2].w + self.learning_rate*np.dot(self.layer_stack[2].dw,(self.layer_stack[3].dx*self.layer_stack[4].dl)).T
        self.layer_stack[2].b = self.layer_stack[2].b + self.learning_rate*(self.layer_stack[3].dx*self.layer_stack[4].dl)
        self.layer_stack[0].w = self.layer_stack[0].w + self.learning_rate*np.dot(self.layer_stack[1].dx*np.dot((self.layer_stack[3].dx*self.layer_stack[4].dl),self.layer_stack[2].dx).T,self.layer_stack[0].dw.T)
        self.layer_stack[0].b = self.layer_stack[0].b + self.learning_rate*self.layer_stack[1].dx*np.dot((self.layer_stack[3].dx*self.layer_stack[4].dl),self.layer_stack[2].dx).T
        print("weights and bias are updated")
        
    def train(self,data_x,data_y):
        for x,y in zip(data_x,data_y):
            self.forwardPass(x,y)
            self.backwardPass()
    
    def sketch(self,data_x,data_y):
        print("this is the evaluated graph")
        for x,y in zip(data_x,data_y):
            self.forwardPass(x,y)
            if round(self.layer_stack[4].pred[0,0]) == 0:
                plt.scatter(x[0][0],x[1][0],color = "yellow")
            else:
                plt.scatter(x[0][0],x[1][0],color = "green")
        plt.show()
            
    def accuracy(self,data_x,data_y):
        right = 0
        wrong = 0
        for x,y in zip(data_x,data_y):
            self.forwardPass(x,y)
            if round(self.layer_stack[4].pred[0,0]) == self.layer_stack[4].y[0,0]:
                right += 1
            else:
                wrong += 1
                
        print("accuracy = ",right/(right+wrong))
if __name__ == "__main__":   
    data = pd.read_csv('log_data.csv')
    data_x = np.array(data.iloc[:,[2,3]].values,dtype = np.float64).reshape(400,2,1)
    data_y = np.array(data.iloc[:,[4]].values,dtype = np.float64).reshape(400,1,1)
    mean = np.mean(data_x,axis = 0)
    sd = np.var(data_x,axis = 0)**.5
    data_x[:,0] = (data_x[:,0]-mean[0])/sd[0]
    data_x[:,1] = (data_x[:,1]-mean[1])/sd[1] 
    my_model = Network(2,6,1,.042)
    my_model.train(data_x,data_y)      
        
    
    #x = np.array(my_model.loss_list)
    #plt.plot(x[:,0,0])
    #plt.show()    
    
    
    #my_model.sketch(data_x,data_y)
    my_model.accuracy(data_x,data_y)