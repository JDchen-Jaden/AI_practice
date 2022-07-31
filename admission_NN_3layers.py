from importlib.util import LazyLoader
from xml.sax.xmlreader import InputSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Implement a simple NN with structure: input(6)-hidden(2)-out(1)
# Two steps:
#     Feedforward:
#         x -- input
#         w_i2h -- input to hidden weights   (n by m); n: input nodes, m: hidden nodes
#         h_in -- input to hidden layer: x*wi2h
#         h-out -- output of hidden layer; Sigmiod(h_in)
#         w_h2o -- hidden to output weidhts  (m by p); n: input nodes, m: hidden nodes
#         out_in -- input to output layer
#         out -- output; Sigmoid AF
#     Backpropagation
#         error  -- prediction error: label - prediction
#         out_et   -- output error term: error*Sigmoid'(out)
#         error_h  -- hidden error: out_et*w_h2o
#         h_et     -- hidden error output: error_h*Sigmoid'(h_out)
#         del_w_h2o -- step size of w_h2o: out_et*h_hout
#         del_w_i2h -- stpe size of w_i2h: h_et*x
# Gradient decent 
#       dEdw_h2o =  out_et*h_out
#       dEhdw_i2h =  h_et*x

# functions
def data_plot(data):
    X = np.array(data[['gre','gpa']])
    y = np.array(data['admit'])
    color = ['red' if i == 0 else 'blue' for i in y]
    plt.scatter(X[:,0],X[:,1], color = color)
    plt.xlabel('GRE')
    plt.ylabel('GPA')

# def  trian_test_split(data,train_size):
#     y = data['admit']
#     x = data.drop('admit', axis = 1)
#     indices = np.random.permutation(y.shape[0])
#     train_indices = indices[:int(len(y)*train_size)]
#     test_indices = indices[int(len(y)*train_size):]
#     print(len(train_indices))
#     print(len(test_indices))
#     return x.iloc[train_indices],x.iloc[test_indices],y.iloc[train_indices],y.iloc[test_indices]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def output_error_term(y,y_hat):
    return (y-y_hat)*sigmoid_prime(y_hat)

def hidden_error_term(out_et,w_h2o,h_out):
    return np.dot(out_et,w_h2o)*sigmoid_prime(h_out)

def train_nn(features, label, epochs, learnrate,n_hidden):
    np.random.seed(42)
    n_records, n_features = features.shape
    # Initialize weights
    w_i2h = np.random.normal(scale=1 / n_features**.5, size=(n_features,n_hidden))
    w_h2o = np.random.normal(scale=1 / n_features**.5, size=n_hidden)
    for i in range(epochs):
        del_w_i2h = np.zeros(w_i2h.shape)
        del_w_h2o = np.zeros(w_h2o.shape)
        for x, y in zip(features.values, label):
            h_out = sigmoid(np.matmul(x,w_i2h))
            y_hat = sigmoid(np.dot(h_out,w_h2o))
            out_et = output_error_term(y,y_hat)
            h_et = hidden_error_term(out_et,w_h2o,h_out)
            del_w_i2h = h_et* x[:, None]
            del_w_h2o = out_et*h_out
        w_i2h += learnrate*del_w_i2h/n_records
        w_h2o += learnrate*del_w_h2o/n_records
    return w_i2h,w_h2o
# load and plot data
data = pd.read_csv('student_data.csv')
print(data.head(10))
# plot data catagorized by rank 
data_rank1 = data[data['rank']==1]
data_rank2 = data[data['rank']==2]
data_rank3 = data[data['rank']==3]
data_rank4 = data[data['rank']==4]

plt.figure(1)
data_plot(data)
plt.title('all')
plt.show(block = False)

plt.figure(2)
plt.title('rank 1')
data_plot(data_rank1)
plt.show(block = False)

plt.figure(3)
plt.title('rank 2')
data_plot(data_rank2)
plt.show(block = False)

plt.figure(4)
plt.title('rank 3')
data_plot(data_rank3)
plt.show(block = False)

plt.figure(5)
plt.title('rank 4')
data_plot(data_rank4)
plt.show(block = False)

# One-hot encoding 
one_hot_data = pd.get_dummies(data, columns = ['rank'])
print(one_hot_data.head(10))
# normalize data & split data into train and test
processed_data = one_hot_data.copy()
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0

# x_train,x_test,y_train,y_test = trian_test_split(data, 0.9)
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
print(f'Length of trainning sample: {len(train_data)}')
print(f'Length of testing sample: {len(test_data)}')
# creat input and label
input = train_data.drop('admit', axis=1)
label = train_data['admit']
input_test = test_data.drop('admit', axis=1)
label_test = test_data['admit']

n_h = 2         # hidden layer
epochs = 1000
learnrate = 0.005

w_i2h,w_h2o = train_nn(input, label, epochs, learnrate,n_h)

# Calculate accuracy on test data
hidden = sigmoid(np.dot(input_test, w_i2h))
print(hidden)
out = sigmoid(np.dot(hidden, w_h2o))
print(out)
predictions = out > 0.5
accuracy = np.mean(predictions == label_test)
print("Prediction accuracy: {:.3f}".format(accuracy))