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
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim

os.path.dirname(os.path.abspath(__file__))
print(os.listdir("input"))

# parameters = dict(
#     lr = [0.01, 0.001]
#     ,batch_size = [ 100, 1000]
#     ,shuffle = [True, False])

parameters = dict(
    lr = [0.01, 0.001]
    ,batch_size = [ 100, 1000]
    ,shuffle = [True, False])

param_values = [v for v in parameters.values()]



def correct(preds, labels):
    v=torch.eq(preds.argmax(dim=1),labels,out=torch.tensor([]))
    return v.sum().item()

class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
    
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
    
        self.network = None
        self.loader = None
        self.tb = None
        

        
class mnist(Dataset):
    
    def __init__(self) : 
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




#load dataset 
dataset= mnist()

#prepare dataloader


# data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
# images, labels = next(iter(data_loader))
# grid = torchvision.utils.make_grid(images.view(10,1,28,28))

for lr, batch_size, shuffle in product(*param_values):
    comment = f' batch_size={batch_size}, lr={lr} , shuffle={shuffle}'
        
    # initialize NN    
    network = Network()    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    optimizer = optim.SGD(network.parameters(), lr=lr)
    
    
    
    tensorb = SummaryWriter(comment=comment)
    # tensorb.add_image("images", grid)
   # tensorb.add_graph(network, images.float())


    for epochs in range(30):
    
        total_loss = 0
        total_correct = 0
        
        
        #take all batch
        for batch in data_loader :
            images, labels = batch
            
            #runs the batch in the NN
            preds=network(images.float())
            
            
            #calculate Loss
            loss = F.cross_entropy(preds,labels)
            optimizer.zero_grad()
            
            #BackProp
            loss.backward()
            
            #update weights
            optimizer.step()
        
        
            total_loss += loss.item() * batch_size
            total_correct += correct(preds,labels)
        
        tensorb.add_scalar('Loss', total_loss, epochs)
        tensorb.add_scalar('Number Correct', total_correct, epochs)
        tensorb.add_scalar('Accuracy', total_correct/len(dataset))
        
        for name, weight in network.named_parameters():
            # tensorb.add_histogram('conv1d bias', network.conv1d.bias, epochs)
            tensorb.add_histogram(name, weight, epochs)
            tensorb.add_histogram(f'{name}.grad', weight.grad, epochs)
        
        
        print("epoch:",epochs ,"/  total_correct:", total_correct, "/  Loss:",total_loss )

    tensorb.close()


    def get_all_preds(model, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images.float())
            all_preds = torch.cat((all_preds, preds),dim=0)
            return all_preds
  
    with torch.no_grad():    
        prediction_loader = DataLoader(dataset=dataset,batch_size=42000,shuffle=False)
        train_preds = get_all_preds(network, prediction_loader)
    
    preds_labels = torch.stack((dataset.y, train_preds.argmax(dim=1)),dim=1)
        
    
    conf_matrix = torch.zeros(10,10,dtype=torch.int32)
    for i in preds_labels:
        l, p = i.tolist()
        conf_matrix[l,p] = conf_matrix[l,p] + 1

# for a in range(10):
    
    # sample=next(i)
    # image, label = sample
    # print(a)
    # print(label)
    # axs[a].imshow(np.array(image.data.tolist()).reshape(28,28))
   
    # pred=network(image.view(1,1,-1))
    # pred.argmax(dim=1)