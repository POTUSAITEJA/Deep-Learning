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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

input_size   = 784

model = nn.Sequential(nn.Linear(input_size, 400),
                      nn.ReLU(),
                      nn.Linear(400,200),
                      nn.ReLU(),
                      nn.Linear(200, 100),
                      nn.ReLU(),
                      nn.Linear(100,10))
                      #nn.Softmax(dim=1))
print(model)

epochs = 6
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for e in range(epochs):
    print(f"Epoch: {e+1}/{epochs}")
    losses = list()
    accuracies = list()

    for i, (images, labels) in enumerate(iter(trainloader)):

        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)
        model.train()
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        losses.append(loss.item())
        accuracies.append(labels.eq(output.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {e+1}, train loss: {torch.tensor(losses).mean():.4f}',end=', ')
    print(f'train accuracy: {torch.tensor(accuracies).mean():.4f}')


    model.eval()
    lossess = list()
    accuraciess = list()
    for i, (images, labels) in enumerate(iter(testloader)):

        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)
        with torch.no_grad():
          output = model.forward(images)   # 1) Forward pass
          loss = criterion(output, labels) # 2) Compute loss
          lossess.append(loss.item())
          accuraciess.append(labels.eq(output.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {e+1}, test loss: {torch.tensor(lossess).mean():.4f}',end=', ')
    print(f'test accuracy: {torch.tensor(accuraciess).mean():.4f}')
    #model.train()
