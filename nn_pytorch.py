import torch 
import numpy as np
import matplotlib.pyplot as plt
import helper_pytorch

# downlaod MINIST datasets, then create training and test datasets
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

from torchvision import datasets, transforms
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
# load training data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# convert traning data to be a iterator
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))  # (64,1,28,28) tensor: 64 images, 1-grey scacle, 28x28 pixels
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show(block = False)


# some functions 
def sigmoid(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    # dim = 1: sum across column
    # (64x10)/(64,) 
    # view(-1,1) convert (64,) to (64,1) then apply broadcasting
    return torch.exp(x)/torch.sum(torch.exp(x),dim=1).view(-1,1)  
# Fully-connected NN or dense NN requires input as 1D vector
# Flattening 
features = images.view(images.shape[0],-1)  #-1 here is used to ensure the number of elements match to the original
n_input = features.shape[1]
n_hidden = 256
n_output = 10
w1 = torch.randn(n_input,n_hidden)
w2 = torch.randn(n_hidden,n_output)
b1 = torch.randn((1,n_hidden))
b2 = torch.randn((1,n_output))
h = sigmoid(torch.matmul(features,w1)+b1)   # 64x256 + 1x256  PyTorch broadcasting
output = sigmoid(torch.matmul(h,w2)+b2)  # 64x10
# convert to probability distribution
p = softmax(output)
print(p.shape)
print(p.sum(dim = 1))

#--------------------------------------------------------------------------------------------------
# Duplicate above approch using Pytorch 
from torch import nn 
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784,256) # input to hidden linear transformation
        self.output = nn.Linear(256,10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):   # must have with PyTorch; Sequential!!!
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        x = self.output(x)
        
        return x
    # A more concise way
    #def __init__(self):
    #    super().__init__()
    #    self.hidden = nn.Linear(784, 256)
    #    self.output = nn.Linear(256, 10)
        
    #def forward(self, x):
    #    x = F.sigmoid(self.hidden(x))
    #    x = F.softmax(self.output(x), dim=1)

# A more practical way 
class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)  # fc - fully connected
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x

model = Network2()
print(model)

# forward pass

ps = model.forward(images[0,:].view(1,-1))
helper_pytorch.view_classify(images[0].view(1,28,28), ps)
plt.show(block = False)


#-----------------------------------------------------------------------------
# Using nn.Sequential

# define hyperparameters
input_size = 784
hidden_size = [128,64]
output_size = 10
# feedforward
model2 = nn.Sequential(nn.Linear(input_size,hidden_size[0]),
                       nn.ReLU(),
                       nn.Linear(hidden_size[0],hidden_size[1]),
                       nn.ReLU(),
                       nn.Linear(hidden_size[1],output_size),
                       nn.Softmax(dim=1))   

ps2 = model2.forward(images[0,:].view(1,-1))
helper_pytorch.view_classify(images[0].view(1,28,28), ps)
plt.show()