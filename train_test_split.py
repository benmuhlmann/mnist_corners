# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:43:36 2020

@author: benmu
"""

import os
import random 
from shutil import copyfile

def train_test_split(input_path,test_path,train_path,test_proportion=0.2):
    #begin error checking
    if not (1>=test_proportion>=0):
        print('enter a proportion between 0 and 1')
        return
    
    path_error=False
    
    if not os.path.isdir(input_path):
        print('The input path is invalid')
        path_error=True
        
    if not os.path.isdir(train_path):
        print('The train path is invalid')
        path_error=True
        
    if not os.path.isdir(test_path):
        print('The test path is invalid')
        path_error=True
        
    if (path_error):
        return
    #end error checking
    
    all_ims=os.listdir(input_path)
   
    random.shuffle(all_ims)
    
    num_test=int(test_proportion*len(all_ims))
    test_ims=all_ims[0:num_test]
    train_ims=all_ims[num_test:]
    
    #copy to test
    [copyfile(os.path.join(input_path,image_file),os.path.join(test_path,image_file)) for image_file in test_ims]
    
    #copy to train
    [copyfile(os.path.join(input_path,image_file),os.path.join(train_path,image_file)) for image_file in train_ims]
    
    
    