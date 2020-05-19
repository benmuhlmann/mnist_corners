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
            self.conv2=nn.Conv2d(4,8,5)
            self.fc=nn.Linear(8*9*9,2)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.fc(x.view(-1,8*9*9))
        return x

hyperparams={'batch_size':12, 'epochs':5, 'learning_rate':0.002, 'num_workers':2}

train_path='train/'
test_path='test/'

TRANSFORM=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])

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

num_train=len(train_data)/
num_test=len(test_data)

num_train_batches=len(train_data_loader)
num_test_batches=len(test_data_loader)

#model
my_model=SimpleCNN()
optimizer=torch.optim.Adam(my_model.parameters(), lr=hyperparams['learning_rate'])
loss_fn=nn.CrossEntropyLoss()

#training
for epoch in range(hyperparams['epochs']):
    #train
    print('EPOCH: {}' .format(epoch)) 
    current_train_loss=0.0
    train_accuracy=0.0
    my_model.train()
    for inputs, classes in test_data_loader:
        optimizer.zero_grad()
        score=my_model(inputs)
        loss=loss_fn(score, classes)
        loss.backward()
        optimizer.step()
        current_train_loss += loss.detach().numpy()
        train_accuracy += sum(score.argmax(dim=1) == classes).numpy()
    
    #test
    current_test_loss=0.0
    test_accuracy=0.0
    my_model.eval()
    with torch.no_grad():
        for inputs, classes in test_data_loader:
            score=my_model(inputs)
            loss=loss_fn(score, classes)
            current_test_loss += loss.detach().numpy()
            test_accuracy += sum(score.argmax(dim=1) == classes).numpy()
    
    #statistics
    train_loss = current_train_loss/num_train_batches
    test_loss  = current_test_loss/num_test_batches
    train_accuracy /= num_train
    test_accuracy /= num_test

    print( '  loss (train, test): {:.4f}, {:.4f}'.format(train_loss, test_loss))
    print( '  accuracy (train, test): {:.4f}, {:.4f}'.format(train_accuracy, test_accuracy))
    