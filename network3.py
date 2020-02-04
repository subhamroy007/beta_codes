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
        
    def backwardPass(self):
        self.layer_stack[0].backwardVal()
        self.layer_stack[1].backwardVal()
        self.layer_stack[2].backwardVal()
        self.layer_stack[3].backwardVal()
        self.layer_stack[4].calculateGradient()
        
        
        temp1 = np.dot(self.layer_stack[3].dx,self.layer_stack[4].dl)
        temp2 = self.layer_stack[1].dx*np.dot(self.layer_stack[2].dx.T,temp1)
        self.layer_stack[2].w = self.layer_stack[2].w - self.learning_rate*np.dot(self.layer_stack[2].dw,temp1.T).T
        self.layer_stack[2].b = self.layer_stack[2].b - self.learning_rate*temp1
        self.layer_stack[0].w = self.layer_stack[0].w - self.learning_rate*np.dot(temp2,self.layer_stack[0].dw.T)
        self.layer_stack[0].b = self.layer_stack[0].b - self.learning_rate*temp2
        print("weights and bias are updated")
        
    def train(self,data_x,data_y):
        count = 1000
        for x,y in zip(data_x,data_y):
            #if count == 0:
             #   break
            #count -= 1
            self.forwardPass(x,y)
            self.backwardPass()
            #break
    
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
        print(right/(right+wrong))
                
        

if __name__ == "__main__":
    dataset = pd.read_csv("Churn_Modelling.csv")
    x = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, [13]].values
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
    labelencoder_X_2 = LabelEncoder()
    x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [1])
    onehotencoder2 = OneHotEncoder(categorical_features = [0])
    x = onehotencoder.fit_transform(x).toarray()
    y = onehotencoder2.fit_transform(y).toarray()
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)
    
    x = np.array(x,dtype = np.float64).reshape(10000,12,1)
    y = np.array(y,dtype = np.float64).reshape(10000,2,1)
    
    
    my_model = Network(12,24,2,.12)
    my_model.train(x,y)
    my_model.accuracy(x,y)
    
    
    
    
