#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:16:20 2020

@author: aktasos
"""
import torch
import time
from collections import namedtuple
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from analytics import correct
import json

class Run():
    
    def __init__(self):
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        self.run_duration = None
        self.network = None
        self.data_loader = None
        self.tb = None
        
    #params = dictionary of parameters for DataLoader and optim
    # dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
    
    def run_parameters(self, params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
    
    def run_begin(self, run, network, data_loader):
        
        self.run_count += 1
        self.run_start_time = time.time()
        self.run_params = run
        self.tb = SummaryWriter(comment=f'-{run}')
        self.network = network
        self.data_loader = data_loader
        
        # images, labels = next(iter(self.data_loader))
        # grid = torchvision.utils.make_grid(images)
    
        # self.tb.add_image('images', grid)
        # self.tb.add_graph(self.network, images)
    
    def save(self, fileName):
    
        pd.DataFrame.from_dict( self.run_data, orient='columns').to_csv(f'{fileName}.csv')
    
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        PATH = f'./trained_models/{self.run_params}.pth'
        torch.save(self.network.state_dict(), PATH)
    
class Epochs(Run):
    
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        self.epoch_end_time = None
        self.epoch_duration = None
        
        
        
    def start_epoch(self):
        
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        
    def end_epoch(self, r):
        
        self.epoch_duration = time.time() - self.epoch_start_time
        r.run_duration = time.time() - r.run_start_time
    
        loss = self.epoch_loss / len(r.data_loader.dataset)
        accuracy = self.epoch_num_correct / len(r.data_loader.dataset)
        
        
        
        r.tb.add_scalar('Loss', loss, self.epoch_count)
        r.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
    
        for name, param in r.network.named_parameters():
            r.tb.add_histogram(name, param, self.epoch_count)
            r.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        results = dict()
        results["run"] = r.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = self.epoch_duration
        results['run duration'] = r.run_duration
        for k,v in r.run_params._asdict().items(): results[k] = v
        r.run_data.append(results)
        
        df = pd.DataFrame.from_dict(r.run_data, orient='columns')
      
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
        
    def track_loss(self, loss, batch_size):
        self.epoch_loss += loss.item() * batch_size
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += correct(preds, labels)
        
   
        
    