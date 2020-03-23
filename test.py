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
        # Function.
        
        self.fc1 = nn.Linear(in_features= 1*784, out_features=588)
        self.fc2 = nn.Linear(in_features= 588, out_features=392)
        self.fc3 = nn.Linear(in_features= 392, out_features=196)
        self.out = nn.Linear(in_features= 196, out_features=10)
        

    def forward(self,x):
        
        #layer 1 
        x = self.fc1(x) 
        x = F.relu(x)
        print(x)
        #layer 2 
        x = self.fc2(x) 
        x = F.relu(x)
        print(x)
        #layer 3 
        x = self.fc3(x) 
        x = F.relu(x)
        print(x)
        #layer 4 
        x = self.out(x) 
        x = F.relu(x)
        print(x)
        
        x = F.softmax(x, dim=1)
        print(x)
        return x
    
network = Network()
dataset= mnist()
data_loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True)
i = iter(dataset)
# fig, axs = plt.subplots(10)
batch = next(iter(data_loader))
images, labels = batch
pred=network(images.float())
pred.argmax(dim=1)