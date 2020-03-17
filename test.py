#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:05:43 2020

@author: aktasos
"""

class test1():
    
    def __init__(self):
        self.a=10
        
class test2(test1):
    def __init__(self):
        self.b = test1.a
    
    def incremant (a):
        test1.run_duration += 1