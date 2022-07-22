import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read csv data as Pandas DataFrame
data = pd.read_csv("data.csv",names=['X1','X2','Y'])
X = data[['X1','X2']].values # Note, this generate a copy of slicing; But .loc won't 
print(type(X))
Y = data['Y'].values
np.random.seed(42)
# Some functions

def step_func(x):
    if x >= 0:
        return 1
    return 0

def prediction(X,W,b):
    return step_func((np.matmul(X,W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []

    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return np.array(boundary_lines)
    
# for i in range(len(X)):
lines = trainPerceptronAlgorithm(X, Y, learn_rate = 0.01, num_epochs = 25)
boundary = lines.reshape(lines.shape[0], lines.shape[1])
print(boundary.shape)
# Generate plot
color  = ['red' if i ==1 else 'blue' for i in Y]
plt.scatter(X[:,0],X[:,1], color = color)

c_red = np.linspace(1,0,len(boundary))
print(c_red.size)
plt.xlim(0,1)
plt.ylim(0,1)
x = np.linspace(0,1,100)
for i in range(len(boundary)):
    print(i)
    k = boundary[i][0]
    b = boundary[i][1]
    y = k*x + b 
    plt.plot(x,y, color = (0,c_red[i], 0))
plt.show()

