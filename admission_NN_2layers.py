import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def cross_entropy(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

# dE/dW = -(y-y_hat)*activation_func_prime*x 
# w = w + dE/dW
# error term: delta = (y-y_hat)*activation_func_prime
def error_term(x,y,y_hat):
    return (y-y_hat)*sigmoid_prime(x)

def train_nn(features, label, epochs, learnrate):
    np.random.seed(42)
    n_records, n_features = features.shape
    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    del_w = np.zeros(weights.shape)
    for i in range(epochs):
        for x, y in zip(features.values, label):
            y_hat = sigmoid(np.matmul(x,weights))
            ce = cross_entropy(y,y_hat)
            delta = error_term(x,y,y_hat)
            del_w = delta*x
        weights += learnrate*del_w/n_records
    return weights
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

epochs = 1000
learnrate = 0.5
weights = train_nn(input, label, epochs, learnrate)
print(input.shape)
# Calculate accuracy on test data
test_out = sigmoid(np.dot(input_test, weights))
print(test_out)
predictions = test_out > 0.5
accuracy = np.mean(predictions == label_test)
print("Prediction accuracy: {:.3f}".format(accuracy))