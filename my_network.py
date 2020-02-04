#in this version og neural network sgc is implemented along mith multiclass
#classification using softmax activation and categorical cross entropy loss


#import necessery libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


#uniform layer
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




#categorical cross entropy loss node
class CategoricalCrossEntropy:
    def __init__(self):
        self.y = None
        self.pred = None
        self.l = None
        self.dl = None
    
    def calculateLoss(self,y,pred):
        self.y = y
        self.pred = pred
        self.l = -y*np.log10(pred)
        return self.l

    def calculateGradient(self):
        self.dl = -self.y * (self.pred ** -1)



#creating softmax activation node
        
class Softmax:
    def __init__(self):
        self.x = None
        self.dx = None
        self.forward_val = None
    
    def forwardVal(self,x):
        self.x = x
        self.forward_val = np.zeros(x.shape,dtype = np.float64)
        
        for i in range(x.shape[0]):
            self.forward_val[i,0] = np.exp(x[i,0])/np.exp(x).sum()
        #self.forward_val = self.forward_val / self.forward_val.sum()
        
        return self.forward_val

    def backwardVal(self):
        #self.dx = np.zeros(self.forward_val.shape,dtype = np.float64)
        self.dx = np.zeros((self.forward_val.shape[0],self.forward_val.shape[0]),dtype = np.float64)
        for i in range(self.dx.shape[0]):
            for j in range(self.dx.shape[0]):
                if i == j:
                    self.dx[i,j] = self.forward_val[i,0]*(1-self.forward_val[j,0])
                else:
                    self.dx[i,j] = -self.forward_val[i,0]*self.forward_val[j,0]
        #self.dx = (temp.sum(axis = 1).reshape(self.dx.shape))
        
        
        
class Network:
    def __init__(self,input_size,layer_size,output_size):
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_size = output_size
        self.layer_stack = []
        self.loss_list = []
        self.fitness = None
        
        
        #initializing all the neccessery computation nodes 
        self.layer_stack.append(Uniform(self.init_weight(self.layer_size,self.input_size),self.init_bias(self.layer_size,1)))
        self.layer_stack.append(Relu())
        self.layer_stack.append(Uniform(self.init_weight(self.output_size,self.layer_size),self.init_bias(self.output_size,1)))
        self.layer_stack.append(Softmax())
        self.layer_stack.append(CategoricalCrossEntropy())

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
        #print("first sum = ",temp)
        temp = self.layer_stack[1].forwardVal(temp)
        #print("relu = ",temp)
        temp = self.layer_stack[2].forwardVal(temp)
        #print("second sum = ",temp)
        temp = self.layer_stack[3].forwardVal(temp)
        #print("softmax = ",temp)
        temp = self.layer_stack[4].calculateLoss(sample_y,temp)
        self.loss_list.append(temp)
        #print("loss = ",temp)
        
        
            
    def generate(self,parent,mu):

        w0 = self.layer_stack[0].w
        w0 = w0.reshape(w0.shape[0]*w0.shape[1],)
        b0 = self.layer_stack[0].b
        b0 = b0.reshape(b0.shape[0]*b0.shape[1],)
        w1 = self.layer_stack[2].w
        w1 = w1.reshape(w1.shape[0]*w1.shape[1],)
        b1 = self.layer_stack[2].b
        b1 = b1.reshape(b1.shape[0]*b1.shape[1],)
        my_weights = np.concatenate([w0,b0,w1,b1])
        w0 = parent.layer_stack[0].w
        w0 = w0.reshape(w0.shape[0]*w0.shape[1],)
        b0 = parent.layer_stack[0].b
        b0 = b0.reshape(b0.shape[0]*b0.shape[1],)
        w1 = parent.layer_stack[2].w
        w1 = w1.reshape(w1.shape[0]*w1.shape[1],)
        b1 = parent.layer_stack[2].b
        b1 = b1.reshape(b1.shape[0]*b1.shape[1],)
        patner_weights = np.concatenate([w0,b0,w1,b1])
        new_weights = np.concatenate([my_weights[:round(len(my_weights)/4)],patner_weights[round(len(my_weights)/4):round(len(my_weights)/2)],my_weights[round(len(my_weights)/2):round(len(my_weights)*3/4)],patner_weights[round(len(my_weights)*3/4):]])   
        for a in range(len(new_weights)):
            if  random.random() <= mu:
                new_weights[a] += np.random.randn()
        #new_weights = []
        #c = -1
        #for x,y in zip(my_weights,patner_weights):
         #   c += 1
          #  k1 = random.random()
           # k2 = random.random()
           # new_weights.append(x*.4+y*(1-.6))
            #if k2 <= mu:
             #   new_weights[c] += np.random.uniform(-1.0, 1.0, 1)
         
        
        new_weights = np.array(new_weights,dtype = np.float64)
        #print(len(new_weights))
        ww0 = new_weights[0:len(w0)].reshape(self.layer_stack[0].w.shape)
        bb0 = new_weights[len(w0):len(w0)+len(b0)].reshape(self.layer_stack[0].b.shape)
        ww1 = new_weights[len(w0)+len(b0):len(w0)+len(b0)+len(w1)].reshape(self.layer_stack[2].w.shape)
        bb1 = new_weights[len(w0)+len(b0)+len(w1):].reshape(self.layer_stack[2].b.shape)
        
        child = Network(12,40,2)
        child.layer_stack[0] = Uniform(ww0,bb0)
        child.layer_stack[2] = Uniform(ww1,bb1)
        
        return child
        
        
            
    def run(self,data_x,data_y):
        right = 0
        wrong = 0
        for x,y in zip(data_x,data_y):
            self.forwardPass(x,y)
            if round(self.layer_stack[4].pred[0,0]) == self.layer_stack[4].y[0,0]:
                right += 1
            else:
                wrong += 1
        self.fitness = 100 * right/(right+wrong)
                
        

