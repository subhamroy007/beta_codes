#let us optimize the cos function using gradient with momentum
import math
import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return np.sin(x)


def grad(x):
    return np.cos(x)



input_x = [x/100 for x in range(0,314,1)]


output_x = func(input_x)
grad_x = grad(input_x)
noise_x = []
for i in range(len(output_x)):
    noise_x.append(output_x[i]+np.random.randn()/10)

plt.plot(input_x,output_x,color = "blue")
plt.scatter(input_x,noise_x,color = "green")
#plt.plot(grad_x,color = "red")
plt.show()

def mainn():
    x = 0.0
    v = 0.0
    beta = .9
    alpha = .1
    x_prev = 0
    while True:
        v = beta * v + (1-beta)*grad(x)
        x_prev = x
        x += alpha*v
        print("updated value of x = ",x)
        if x == x_prev:
            print("found optimal value")
            break
    
v = [0]*len(output_x)
v[0] = noise_x[0]
v[1] = noise_x[1]
for i in range(2,len(noise_x)):
    v[i] = (noise_x[i]+noise_x[i-1]+noise_x[i-2])/3

noise_x = v

plt.plot(output_x,color = "blue")
plt.plot(noise_x,color="green")
plt.show()




