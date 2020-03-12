#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:55:53 2020

@author: aktasos
"""

# Kaggle_digit_recogniser
import os
from tensorfromdata import TensorDataset
from network import Network
from torch.utils.data import DataLoader
from run_options import RunOptions

os.path.dirname(os.path.abspath(__file__))
#parameters = dictionary of parameters for DataLoader and optim (dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
parameters = dict(
    lr = [0.01, 0.001]
    ,batch_size = [1000, 2000]
    ,shuffle = [True, False])

r = RunOptions()
runs = r.run_parameters(parameters)

data_path = "input/train.csv"
train_set = TensorDataset(data_path)
network = Network()


data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
optimizer = optim.SGD(network.parameters(), lr=lr)
tensorb = SummaryWriter(comment=comment)