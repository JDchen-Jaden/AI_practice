from re import T
import torch 
import helper_pytorch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

images, labels = next(iter(trainloader))
helper_pytorch.imshow(images[1,:])
plt.show(block = False)

# build NN
model = nn.Sequential(nn.Linear(784,256), nn.ReLU(),
                      nn.Linear(256,128), nn.ReLU(),
                      nn.Linear(128,64),  nn.ReLU(),
                      nn.Linear(64,10), nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.003)
epochs = 5

for i in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
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
helper_pytorch.view_classify(img.view(1,28,28), ps, version='Fashion')
plt.show()