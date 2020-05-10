# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:05:38 2020

@author: benmu
"""
import random
import cv2
import os
from cornerize_cv2 import cornerize

def cornerize_folder(path, proportion):
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
    
 
