import torch 
import numpy as np
import matplotlib.pyplot as plt
import helper_pytorch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

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
# images, labels = next(iter(trainloader))
# images = images.view(images.shape[0], -1)

# build a NN
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
# define loss
criterion = nn.NLLLoss()   # negtive log likelihood loss
optimizer = optim.SGD(model.parameters(), lr=0.003)  # stochastic gradient decent
epochs = 5

for i in range(epochs):
    running_loss = 0
    for images,labels in trainloader:
        images = images.view(images.shape[0],-1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f'Training Loss: {running_loss/len(trainloader)}')

# Test result
images,labels = next(iter(trainloader))
img = images[0].view(1,784)
# turn off graidient (default is True) to speed up
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
helper_pytorch.view_classify(img.view(1,28,28), ps)
plt.show()

