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
        
        self.fc1 = nn.Linear(in_features= 784, out_features=588)
        self.fc2 = nn.Linear(in_features= 588, out_features=392)
        self.fc3 = nn.Linear(in_features= 392, out_features=196)
        self.out = nn.Linear(in_features= 196, out_features=10)
        

    def forward(self,x):
        
        #layer 1 
        x = self.fc1(x) 
        x = F.relu(x)

        #layer 2 
        x = self.fc2(x) 
        x = F.relu(x)

        #layer 3 
        x = self.fc3(x) 
        x = F.relu(x)

        #layer 4 
        x = self.out(x) 
        x = F.relu(x)

        
        x = F.softmax(x, dim=1)
        return x