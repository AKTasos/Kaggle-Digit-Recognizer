#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:59:22 2020

@author: aktasos
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.path.dirname(os.path.abspath(__file__)) 
print(os.listdir("input"))


# train = pd.read_csv("input/train.csv")
# targets_numpy = train.label.values
# features_numpy = train.loc[:,train.columns != "label"].values
# plt.imshow(features_numpy[10].reshape(28,28))
# plt.show()

# targets_tensor = torch.tensor(targets_numpy)
# features_tensor = torch.tensor(features_numpy)
#data = torch.utils.data.DataLoader(features_tensor, batch = 100)

class mnist(Dataset) :
    
    def __init__(self):
        train = pd.read_csv("input/train.csv")
        # train = np.loadtxt("./input/train.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.y = torch.from_numpy(train.label.values)
        self.x = torch.from_numpy(train.loc[:,train.columns != "label"].values).unsqueeze(1)
        self.n_samples=train.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
# create class
class Network(nn.Module):
    def __init__(self):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(Network,self).__init__()
        # Linear function.
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=28 )
        self.fc1 = nn.Linear(in_features=10*757, out_features=280)
        self.out = nn.Linear(in_features=280, out_features=10)

    def forward(self,x):
        
        # 1 hidden layer
        x = self.conv1d(x)
        
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        # 2 hidden layer
        x = self.fc1(x)
        x = F.relu(x)
         
        # 3 output layer
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x
    
network = Network()
dataset= mnist()
data_loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True)
i = iter(dataset)
# fig, axs = plt.subplots(10)
batch = next(iter(data_loader))
images, labels = batch
pred=network(images)
pred.argmax(dim=1)


# for a in range(10):
    
#     sample=next(i)
#     image, label = sample
#     print(a)
#     print(label)
#     axs[a].imshow(np.array(image.data.tolist()).reshape(28,28))
   
#     pred=network(image.view(1,1,-1))
#     pred.argmax(dim=1)