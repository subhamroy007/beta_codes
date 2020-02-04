def loss_func(x):
    #print(x)
    return x**3 - 4*x + 5


def grad(x):
    return 3*x**2 -4


x_prev = None
x = 10
alpha = .014
dx = []
loss = []
i = 0
ifc = 1.25
dfc = .5
while True:
    loss.append(loss_func(x))
    dx.append(grad(x))
    if i != 0:
        if dx[i]*dx[i-1] > 0:
            alpha = min(alpha*ifc,20)
        
        elif dx[i]*dx[i-1] < 0:
            alpha = max(alpha*dfc,.0001)
    x_prev = x
    x -= alpha*dx[i]
    i+=1
    print("new x = ",x)
    if x_prev == x:
        break


import matplotlib.pyplot as plt

plt.plot(loss)  
plt.show()     