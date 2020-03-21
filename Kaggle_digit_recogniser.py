#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:55:53 2020

@author: aktasos
"""

# Kaggle_digit_recogniser
import os
from tensorfromdata import TensorDataset

from torch.utils.data import DataLoader
from run_options import Run, Epochs, FcLayers
import torch.optim as optim
import torch.nn.functional as F

os.path.dirname(os.path.abspath(__file__))

n_in = 784
output = 10
nb_of_layers = 10
act_fonction="relu"

#parameters = dictionary of parameters for DataLoader and optim (dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
parameters = dict(
    lr = [0.01]
    ,batch_size = [1000]
    ,shuffle = [False]
    ,epochs = [20]
    ,nb_of_layers = [x for x in range(4, 20, 4)]
    ,act_fonction=["Relu"])

r = Run()
runs = r.run_parameters(parameters)

data_path = "input/train.csv"
train_set = TensorDataset(data_path)



for run in runs:
    
    layers = FcLayers(n_in, output, run.nb_of_layers, run.act_fonction)
    layers.layer_creation()
    layers.network_creator()
    
    from networkfc import Network
    network = Network()
   
    data_loader = DataLoader(dataset=train_set, batch_size=run.batch_size, shuffle=run.shuffle)
    optimizer = optim.SGD(network.parameters(), lr=run.lr)
    
    r.run_begin(run, network, data_loader)
   
    e = Epochs()
    for epoch in range(run.epochs):
        
       
        e.start_epoch()
        
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
        
            e.track_loss(loss, run.batch_size)
            e.track_num_correct(preds, labels)
            
            
        print("epoch:", e.epoch_count ,"/  total_correct:", e.epoch_num_correct, "/  Loss:", e.epoch_loss )
        e.end_epoch(r)
        
    
    r.end_run()
    r.save('results')