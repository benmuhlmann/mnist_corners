import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import copy

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