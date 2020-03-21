#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:05:43 2020

@author: aktasos
"""
import torch.nn as nn
import torch.nn.functional as F

n_in = 784
output = 10
nb_of_layers = 10
act_fonction="Relu"

class FcLayers():
    
    def __init__(self, n_in, output, nb_of_layers, act_fonction):
        
        self.n_layer = 1
        self.name = (f'fc{self.n_layer}')
        self.in_features = n_in
        self.n_delta = n_in // nb_of_layers
        self.out_features = n_in-self.n_delta
        self.output = output
        self.nb_of_layers = nb_of_layers
        self.layer_list = []
        self.act_fonction="Relu"
        
    def next_layer_parameters(self):
        
        self.n_layer += 1
        self.name = (f'fc{self.n_layer}')
        self.in_features = self.out_features
        self.out_features -= self.n_delta
        
        
    def layer_creation(self):
        for i in range(self.nb_of_layers):
            if i == self.nb_of_layers-1 :
                self.name = "out"
                self.out_features = self.output
            self.layer_list.append((self.name, self.in_features, self.out_features))
            self.next_layer_parameters()
            
            
    def network_creator(self):
        
        text_layer = str()
        text_forward =str()
        l=1
        
        for name, in_feat, out_feat in a.layer_list :
            
            text_layer =text_layer + f'self.{name} = nn.Linear(in_features= {in_feat}, out_features={out_feat}\n'
            text_forward = text_forward + f'#layer {l} \nx=self.{name}(x) \nx=F.{self.act_fonction(x)}\n\n'
            # layer_list.append(f'self.{name} = nn.Linear(in_features= {in_feat}, out_features={out_feat}')
            # text_layer = '\n'.join(layer_list)
            l += 1
        
        
    
        file = open("network_fc_template", "r")
        text = file.read()
        text.replace('layer_line', text_layer)
        file = open(f"Network_{self.nb_of_layers}layers.py", "w")
        file.write(first)
        file.write(first)
        file.write(first)
        
    
    
    
        
 a=fc_layers(n_in, output, nb_of_layers)
a.layer_creation()