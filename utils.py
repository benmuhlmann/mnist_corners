# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:36:53 2020

@author: benmu
"""

import os
import random
from shutil import copyfile, rmtree
import cv2
import numpy as np



def empty_train_test(folder_path):
    if not os.path.isdir(folder_path):
        print('invalid input path')
        
    rmtree(folder_path)
    os.makedirs(folder_path)
    
    
    

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




def cornerize(image):
    
    side_length=image.shape[0]
    half_side=side_length//2
    
    small_image=cv2.resize(image,(half_side,half_side))
    
    black_image=np.reshape(np.zeros(side_length*side_length),(side_length,side_length))
    
    u=np.random.uniform(0,1)
    
    if u<0.25:
        black_image[0:half_side, half_side:] = small_image

    elif u<0.5:
        black_image[0:half_side, 0:half_side] = small_image

    elif u<0.75:
        black_image[half_side:, 0:half_side] = small_image

    else: 
        black_image[half_side:, half_side:] = small_image
        
    return black_image




def cornerize_folder(path, proportion=0.1):
    """ 
        'cornerizes' a proportion of images in the specified folder 
        path: path to the folder containing images you wish to 'cornerize'
        proportion: the proportion of images to be 'cornerized'. Between 0 and 1 
    """    
    
    image_list=os.listdir(path)
    
    num_images=len(image_list)
    
    to_be_cornerized=random.sample(image_list, int(proportion*num_images))
    
    counter=0
    
    for image in to_be_cornerized:
        image_path=os.path.join(path,image)
        image_cv2=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        os.remove(image_path)
        counter+=cv2.imwrite(image_path,cornerize(image_cv2))
    
    return counter==int(proportion*num_images)
    
