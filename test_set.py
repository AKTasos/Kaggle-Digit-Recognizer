#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:12:35 2020

@author: aktasos
"""
from tensorfromdata import TensorDataset
import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np

def submission_file(run,cnnfc):
    data_path_test = "input/test.csv"
    test_set = TensorDataset(data_path_test)
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=False)
    results = []
    # path = "trained_models/Run(lr=0.01, batch_size=10, shuffle=False, epochs=20, nb_of_fclayers=2, kernel_size=4).pth"
    # cnnfc.load_state_dict(torch.load(path))
    
    # results = pd.DataFrame(data=([x for x in range(test_set.x.shape[0])],[]), columns=['ImageId', 'Label'])
    
    for batch in test_loader :
            a+=1            
            print(a)
            #runs the batch in the CNN
            preds = cnnfc(batch.float()).argmax(dim=1)
            results.append(preds)
            
    results = pd.DataFrame(torch.stack(results).view(-1), columns=["Labels"])
    results.index = results.index+1
    results.index.name="ImageId"
    results.to_csv(f'{run}.csv')
        
