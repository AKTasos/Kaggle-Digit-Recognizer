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
        
        self.fc1 = nn.Linear(in_features= 784, out_features=735)
        self.fc2 = nn.Linear(in_features= 735, out_features=686)
        self.fc3 = nn.Linear(in_features= 686, out_features=637)
        self.fc4 = nn.Linear(in_features= 637, out_features=588)
        self.fc5 = nn.Linear(in_features= 588, out_features=539)
        self.fc6 = nn.Linear(in_features= 539, out_features=490)
        self.fc7 = nn.Linear(in_features= 490, out_features=441)
        self.fc8 = nn.Linear(in_features= 441, out_features=392)
        self.fc9 = nn.Linear(in_features= 392, out_features=343)
        self.fc10 = nn.Linear(in_features= 343, out_features=294)
        self.fc11 = nn.Linear(in_features= 294, out_features=245)
        self.fc12 = nn.Linear(in_features= 245, out_features=196)
        self.fc13 = nn.Linear(in_features= 196, out_features=147)
        self.fc14 = nn.Linear(in_features= 147, out_features=98)
        self.fc15 = nn.Linear(in_features= 98, out_features=49)
        self.out = nn.Linear(in_features= 49, out_features=10)
        

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
        x = self.fc4(x) 
        x = F.relu(x)

        #layer 5 
        x = self.fc5(x) 
        x = F.relu(x)

        #layer 6 
        x = self.fc6(x) 
        x = F.relu(x)

        #layer 7 
        x = self.fc7(x) 
        x = F.relu(x)

        #layer 8 
        x = self.fc8(x) 
        x = F.relu(x)

        #layer 9 
        x = self.fc9(x) 
        x = F.relu(x)

        #layer 10 
        x = self.fc10(x) 
        x = F.relu(x)

        #layer 11 
        x = self.fc11(x) 
        x = F.relu(x)

        #layer 12 
        x = self.fc12(x) 
        x = F.relu(x)

        #layer 13 
        x = self.fc13(x) 
        x = F.relu(x)

        #layer 14 
        x = self.fc14(x) 
        x = F.relu(x)

        #layer 15 
        x = self.fc15(x) 
        x = F.relu(x)

        #layer 16 
        x = self.out(x) 
        x = F.relu(x)

        
        
        
        return x.view(1000,10)