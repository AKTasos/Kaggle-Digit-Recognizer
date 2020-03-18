#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(Network,self).__init__()
        # Function.
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4 )
        self.conv2d_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4)
        self.conv2d_3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4)
        self.fc1 = nn.Linear(in_features=12*361, out_features=784)
        self.fc2 = nn.Linear(in_features=784, out_features=784)
        self.out = nn.Linear(in_features=784, out_features=10)

    def forward(self,x):
        
        # 1 hidden layer
        x = self.conv2d_1(x)
        x = F.relu(x)
    
        # 2 hidden layer
        x = self.conv2d_2(x)
        x = F.relu(x)
        
        # 3 hidden layer
        x = self.conv2d_3(x)
        x = F.relu(x)
        
        # 4 hidden layer
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        
        # 5 hidden layer
        x = self.fc2(x)
        x = F.relu(x) 
        
        # 3 output layer
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x