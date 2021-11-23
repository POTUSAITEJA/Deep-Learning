from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import random_split


# Define a transform to normalize the data (Preprocessing)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])

# Download and load the training data
trainset    = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

# Download and load the test data
testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)


input_size   = 784

class Network(nn.Module):
    
    # Defining the layers, 128, 64, 10 units each
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
        
    # Forward pass through the network, returns the output logits
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x

model = Network()
print(model)

# model = nn.Sequential(nn.Linear(input_size, 400),
#                       nn.ReLU(),
#                       nn.Linear(400,200),
#                       nn.ReLU(),
#                       nn.Linear(200, 100),
#                       nn.ReLU(),
#                       nn.Linear(100,10))
#                       #nn.Softmax(dim=1))
# print(model)

epochs = 6
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
img_len=len(trainloader.dataset)

for e in range(epochs):
    print(f"Epoch: {e+1}/{epochs}")
    losses = list()
    accuracies = list()
    train_accuracy = 0

    for i, (images, labels) in enumerate(iter(trainloader)):

        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)

        #model.train()
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        losses.append(loss.item())
        accuracies.append(labels.eq(output.detach().argmax(dim=1)).float().mean())

        _,prediction = torch.max(output.data,1)
        train_accuracy += int(torch.sum(prediction==labels.data))

    train_accuracy = train_accuracy/img_len
    print(train_accuracy)
    print(f'Epoch {e+1}, train loss: {torch.tensor(losses).mean():.4f}',end=', ')
    print(f'train accuracy: {torch.tensor(accuracies).mean():.4f}')



    model.eval()
    lossess = list()
    accuraciess = list()
    for i, (ima, lab) in enumerate(iter(testloader)):

        # Flatten MNIST images into a 784 long vector
        ima.resize_(ima.size()[0], 784)
        with torch.no_grad():
          output = model.forward(ima)   # 1) Forward pass
          losss = criterion(output, lab) # 2) Compute loss
          lossess.append(losss.item())
          accuraciess.append(lab.eq(output.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {e+1}, test loss: {torch.tensor(lossess).mean():.4f}',end=', ')
    print(f'test accuracy: {torch.tensor(accuraciess).mean():.4f}')
    model.train()
