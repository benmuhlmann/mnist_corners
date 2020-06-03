# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 12:02:42 2020

@author: benmu
"""

from utils import empty_train_test, train_test_split, cornerize_folder


# ensure folders are empty
folder_list=['test/0', 'test/1', 'train/0', 'train/1']
[empty_train_test(folder) for folder in folder_list]

#do train test split
train_test_split(input_path='all_0s',test_path='test/0',train_path='train/0')
train_test_split(input_path='all_1s',test_path='test/1',train_path='train/1')

#cornerize '0' folders
cornerize_folder(path='test/0', proportion=0.8)
cornerize_folder(path='train/0', proportion=0.1)
