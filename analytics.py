#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:38:10 2020

@author: aktasos
"""
import torch

def correct(preds, labels):
    c=torch.eq(preds.argmax(dim=1),labels,out=torch.tensor([]))
    return c.sum().item()

def get_all_preds(model, loader):
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images.float())
            all_preds = torch.cat((all_preds, preds),dim=0)
            return all_preds
  
def confusion_matrix():
    with torch.no_grad():    
        prediction_loader = DataLoader(dataset=dataset,batch_size=len(dataset),shuffle=False)
        train_preds = get_all_preds(network, prediction_loader)
    
    preds_labels = torch.stack((dataset.y, train_preds.argmax(dim=1)),dim=1)
        
    
    conf_matrix = torch.zeros(10,10,dtype=torch.int32)
    for i in preds_labels:
        l, p = i.tolist()
        conf_matrix[l,p] = conf_matrix[l,p] + 1