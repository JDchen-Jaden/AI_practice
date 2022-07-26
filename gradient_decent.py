import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot func
def scatter_plot(X,Y):
    color = ['red' if i == 0 else 'blue' for i in Y]
    plt.scatter(X[:,0],X[:,1], color = color, edgecolor = 'k')
def boundary_plot(W,b,i,epochs):
    # boundary line y = kx + d
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.linspace(-0.05,1.05,100)
    k = -W[0]/W[1]
    d = -b/W[1]
    color = 'g'
    lw = 1
    if i == epochs-1:
        color = 'k'
        lw = 2
    plt.plot(x,k*x+d,color = color, linewidth = lw)

# gradient decent func
def sigmoid(x):
    return 1/(1+ np.exp(-x))

def prediction(X,W,b):
    return sigmoid(np.matmul(X,W) + b)

def cross_entropy(y,y_hat):
    return - ( y*np.log(y_hat) + (1-y)*np.log(1-y_hat) )

def update(W,b,X,y,y_hat,alpha):
    W_new = W + alpha*(y-y_hat)*X 
    b_new = b + alpha*(y-y_hat)
    return W_new,b_new 

# tranning func
def training(X,Y,alpha,epochs):
    W = np.random.normal(scale=1 / X.shape[0]**.5, size=X.shape[1])
    #W = np.array([1,1])
    b = 0
    errs = []
    plt.figure(1)
    for i in range(epochs):
        for x, y in zip(X,Y):
            y_hat = prediction(x,W,b)
            err = cross_entropy(y,y_hat)
            W,b = update(W,b,x,y,y_hat,alpha)
        errs.append(err)
        boundary_plot(W,b,i,epochs)
    plt.show(block=False)
    return errs
# Load and plot data
data = pd.read_csv("data_gradient_decent.csv", header=None)
# Convert DataFrame to be numpy array and display data
X = np.array(data[[0,1]])
Y = np.array(data[2])
scatter_plot(X,Y)

np.random.seed(44)
epochs = 100
alpha = 0.01 # learning rate
errs = training(X,Y,alpha,epochs)

plt.figure(2)
plt.plot(errs)
plt.show()