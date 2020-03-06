#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:09:08 2020

@author: aktasos
"""

import numpy as np

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
        train = np.loadtxt("./input/train.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.y = torch.from_numpy(train[:,0])
        self.x = torch.from_numpy(train[:, 1:-1])
        self.n_samples=train.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples