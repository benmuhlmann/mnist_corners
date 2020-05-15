# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:25:53 2020

@author: benmu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self):
            super(SimpleCNN,self).__init__()
            self.conv1=nn.Conv2d(1,4,3)
            self.pool=nn.MaxPool2d(2)
            self.conv3=nn.Conv2d(4,8,5)
            self.fc=nn.Linear(8*9*9,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.fc(x.view(-1,8*9*9))
        return x

hyperparams={'batch_size':12, 'epochs':5, 'learning_rate':0.002, 'num_workers':2}

train_path='train/'
test_path='test/'

TRANSFORM=transforms.Compose([transforms.ToTensor()])

train_data=torchvision.datasets.ImageFolder(root=train_path,transform=TRANSFORM)
train_data_loader=data.DataLoader(train_data, 
                                  batch_size=hyperparams['batch_size'], 
                                  shuffle=True, 
                                  num_workers=hyperparams['num_workers'])

test_data=torchvision.datasets.ImageFolder(root=test_path,transform=TRANSFORM)
test_data_loader=data.DataLoader(test_data, 
                                 batch_size=hyperparams['batch_size'], 
                                 shuffle=True, 
                                 num_workers=hyperparams['num_workers'])

