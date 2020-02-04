def loss_func(x):
    #print(x)
    return x**3 - 4*x + 5


def grad(x):
    return 3*x**2 -4


x_prev = None
x = 10
alpha = 1
loss = []
i = 0
beta = 0.9
egs = 0
while True:
    loss.append(loss_func(x))
    dx = grad(x)
    egs = beta*egs + (1-beta)*dx*dx
    x_prev = x
    x -= alpha*dx/(egs**.5)
    i+=1
    print("new x = ",x)
    if x_prev == x:
        break


import matplotlib.pyplot as plt

plt.plot(loss)  
plt.show()     