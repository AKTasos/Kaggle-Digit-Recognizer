#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:23:09 2020

@author: aktasos
"""
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch


os.path.dirname(os.path.abspath(__file__))
print(os.listdir())
data_path = str


class TensorDataset(Dataset):
    
    def __init__(self, data_path) : 
        data = pd.read_csv(data_path)
        # data = np.loadtxt("./input/data.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.y = torch.from_numpy(data.label.values)
        self.x = torch.from_numpy(data.loc[:,data.columns != "label"].values).unsqueeze(1)
        self.n_samples=data.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples