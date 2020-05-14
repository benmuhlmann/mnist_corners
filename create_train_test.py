# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:00:51 2020

@author: benmu
"""

from cornerize_folder import cornerize_folder
from train_test_split import train_test_split



train_test_split('all_0s','test/0','train/0')
train_test_split('all_1s','test/1','train/1')

cornerize_folder('test/0')
cornerize_folder('train/0')


