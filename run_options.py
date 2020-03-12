#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:16:20 2020

@author: aktasos
"""
import time
from collections import namedtuple
from itertools import product

class RunOptions():
    
    #params = dictionary of parameters for DataLoader and optim
    # dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
    def run_parameters(self, params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
    
class RunManager():
    
    def __init__(self):
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
    
        self.network = None
        self.loader = None
        self.tb = None
    
class Epochs():
    
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None
        self.end_time = None
        self.duration = None
        
        
    def start_epoch(self):
        
        self.start_time = time.time()
        self.count += 1
        self.loss = 0
        
    def end_epoch(self):
        self.duration = time.time() - self.start_time
        RunManager.run_duration = time.time() - self.run_start_time
    
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
    
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
    
        for name, param in self.network.named_parameters():
        self.tb.add_histogram(name, param, self.epoch_count)
        self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        