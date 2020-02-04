import numpy as np
import matplotlib.pyplot as plt

def loss_func(x,y):
    return (x**2/9 + y**2/16)

def grad(x,y):
    return 2*x/9,2*y/16


v1,v2 = 0,0
beta = .8
alpha = .1
x,y = 1,1
loss=[]
for i in range(100):
    loss.append(loss_func(x,y))
    dx,dy = grad(x,y)
    v1 = beta*v1 + (1-beta)*dx
    v2 = beta*v2 + (1-beta)*dy
    x -= (alpha*v1)
    y -= (alpha*v2)
    
    
    
print(x,"  ",y)
plt.plot(loss)