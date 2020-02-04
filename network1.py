#our objective is to develop a fully functional neural network from scratch that supports RELU and SIGMOID 
#activation function and mean square error loss function and back propagets via gradient decent strategy


#importing necessery library
import numpy as np
import random
import func

#defining the linear computational node
class linear_node:
	def __init__(self):
		#user given parameters
		
		self.W = None
		self.X = None
		self.B = None

		#immediate slopes

		self.dw = None
		self.dx = None
		self.db = None

	def forward_val(self,W,X,B):
		self.W = W
		self.X = X
		self.B = B
		self.dw = X.T
		self.dx = W.T
		self.db = np.ones(self.B.shape,dtype = np.float64)
		return np.dot(self.W,self.X) + self.B


	def backward_val(self,pass_val):
		self.db = self.db * pass_val
		self.dw = np.dot(pass_val,self.dw)
		self.dx = np.dot(self.dx,pass_val)
		self.W = self.W - .1*self.dw
		self.B = self.B - .1*self.db



#defining the sigmoid computational node
class sigmoid_node:
	def __init__(self):
		self.X = None
		self.dx = None

	def forward_val(self,X):
		self.X = X
		self.val = (np.ones(self.X.shape,dtype = np.float64) + np.exp(-self.X)) ** -1
		self.dx = self.val * (np.ones(self.X.shape,dtype = np.float64) - self.val)
		return self.val

	def backward_val(self,pass_val):
		self.dx  = self.dx * pass_val


#defining the relu computational node
class relu_node:
	def __init__(self):
		self.X = None
		self.dx = None

	def forward_val(self,X):
		self.X = X
		self.value = np.zeros(self.X.shape,dtype = np.float64)
		self.dx = np.zeros(self.X.shape,dtype = np.float64)
		for i in range(0,self.value.shape[0]):
			for j in range(0,self.value.shape[1]):
				if self.X[i][j] > 0:
					self.value[i][j] = self.X[i][j]
					self.dx[i][j] = 1
		return self.value

	def backward_val(self,pass_val):
		self.dx  = self.dx * pass_val


#defining the mse computational node
class mse_node:
	def __init__(self):
		self.G = None
		self.Y = None
		self.dx = None


	def forward_val(self,G,Y):
		self.G = G
		self.Y = Y
		self.dx = 2 * (self.G-self.Y)
		return (self.Y-self.G) ** 2



#defining the neural network class
class network:
	def __init__(self):
		#initializing the parameters of the network
		self.weight_list = []
		self.bias_list = []
		#self.linear_list = []
		#self.active_list = []
		self.layer_list = []
		self.loss = None
		self.no_of_layers = 0
		self.loss_decider()

	def insert_layer(self,input_size,output_size,activation_function = None):
		self.no_of_layers += 1
		temp = np.ones((output_size,input_size),dtype = np.float64)
		for i in range(output_size):
			for j in range(input_size):
				temp[i][j] = np.random.randn()
		self.weight_list.append(temp)
		temp = np.ones((output_size,1),dtype = np.float64)
		for i in range(output_size):
			temp[i][0] = np.random.randn()
		self.bias_list.append(temp)
		self.layer_list.append(linear_node())
		if activation_function == 0:
			self.layer_list.append(relu_node())
			self.no_of_layers += 1
		elif activation_function == 1:
			self.layer_list.append(sigmoid_node())
			self.no_of_layers += 1
		else:
			pass


	def loss_decider(self):
		self.loss = mse_node()



	def forward_pass(self,x,y):
		for i in range(self.no_of_layers):
			if i % 2 == 0:
				x = self.layer_list[i].forward_val(self.weight_list[round(i/2)],x,self.bias_list[round(i/2)])
			else:
				x = self.layer_list[i].forward_val(x)
		self.loss.forward_val(x,y)
		#print("loss matrix = ",self.loss.forward_val(x,y))

	def backward_pass(self):
		d_original = self.loss.dx
		for i in range(self.no_of_layers-1,-1,-1):
			if i % 2 == 1:
				self.layer_list[i].backward_val(d_original)
				d_original = self.layer_list[i].dx
			else:
				self.layer_list[i].backward_val(d_original)
				d_original = self.layer_list[i].dx
				self.weight_list[round(i/2)] = self.layer_list[i].W
				self.bias_list[round(i/2)] = self.layer_list[i].B
		#print("weights and bias are updated")
		


	def train_network(self,X,Y):
		for i in range(X.shape[0]):
			x = np.array(X[i],dtype = np.float64).reshape(X.shape[1],1)
			y = np.array(Y[i],dtype = np.float64).reshape(Y.shape[1],1)
			print(i+1,x[0][0],y[0][0],self.weight_list[0][0][0],self.bias_list[0][0][0],)
			self.forward_pass(x,y)
			self.backward_pass()
			print(self.loss.dx)

	def estimate(self,x):
		for i in range(self.no_of_layers):
			if i % 2 == 0:
				x = self.layer_list[i].forward_val(self.weight_list[round(i/2)],x,self.bias_list[round(i/2)])
			else:
				x = self.layer_list[i].forward_val(x)
		return x



#print("hello")
nn_obj1 = network()
nn_obj1.insert_layer(1,1)
#x = [[3,1.5],[2,1],[4,1.5],[3,1],[3.5,.5],[2,.5],[5.5,1],[1,1]]
#y = [1,0,1,0,1,0,1,0]
x = []
y = []
x,y = func.func(x,y,10)
x = np.array(x,dtype = np.float64).reshape(10,1)
y = np.array(y,dtype = np.float64).reshape(10,1)
#for i in range(10000):
nn_obj1.train_network(x,y)
