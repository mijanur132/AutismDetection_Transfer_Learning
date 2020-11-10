#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

from ipynb.fs.full.autismDataProcess import data_process
import matplotlib.pyplot as plt
import time
import copy
import os
import sys


# In[20]:


class MiniBatcher(object):
    def __init__(self, batch_size, n_examples, shuffle=True):
        assert batch_size <= n_examples, "Error: batch_size is larger than n_examples"
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.shuffle = shuffle
        print("batch_size={}, n_examples={}".format(batch_size, n_examples))

        self.idxs = np.arange(self.n_examples)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.current_start = 0

    def get_one_batch(self):
        self.idxs = np.arange(self.n_examples)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.current_start = 0
        while self.current_start < self.n_examples:
            batch_idxs = self.idxs[self.current_start:self.current_start+self.batch_size]
            self.current_start += self.batch_size
            yield torch.LongTensor(batch_idxs)


# In[21]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_cuda=0
if torch.cuda.is_available():
    use_cuda=1    
    print("useCuda")

net = models.resnet50(pretrained=True)
net = net.cuda() if device else net
#net


# In[22]:


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net.fc = net.fc.cuda() if use_cuda else net.fc


# In[23]:
learning_rate=0.01
if (len(sys.argv)<2):
    print("Default LR=0.0001, Suggestion- you can provide learning rate as command line argument-->")
    learning_rate=0.0001
else:
    learning_rate=sys.argv[1]
n_epochs = 100
if (len(sys.argv)<3):
    print("Default epochs 50, Suggestion- you can provide N epochs as command line argument-->")
    n_epochs = 50
else:
    n_epochs =sys.argv[2]
n_epochs=int(n_epochs)
learning_rate=float(learning_rate)
print(learning_rate)

print_every = 20
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []


X_train,y_train,X_validation,y_validation,X_test,y_test=data_process()
n_examples = X_train.shape[0]
n_examples_test = X_test.shape[0]
minibatch_size=100
total_step=n_examples/minibatch_size
total_test_step=n_examples_test/minibatch_size
batcher = MiniBatcher(minibatch_size, n_examples)
test_batcher = MiniBatcher(300, n_examples_test)


# In[18]:



optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  #Read further
val_loss = []
val_acc = []
train_loss = []
train_acc = []
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    for train_idxs in batcher.get_one_batch():
        train_idxs = train_idxs#.cuda() if device else train_idxs
        batch_idx=batch_idx+1

        data_, target_ = X_train[train_idxs], y_train[train_idxs]
        data_, target_=data_.to(device), target_.flatten().to(device)
        optimizer.zero_grad()           #read further

        outputs = net(data_.float())
        if(target_.shape[0]<100):
            continue
        loss = criterion(outputs, target_)
        loss.backward()                        #read further
        optimizer.step()                       #read further

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Accuracy: {:.4f}'
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item(),100 * correct / total))
    tempAcc=100 * correct / (total+0.0001)
    train_acc.append(tempAcc)
    train_loss.append(running_loss/(total_step+0.0001))
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(tempAcc):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()                                           #read further
        for test_idxs in test_batcher.get_one_batch():
            data_t, target_t=X_test[test_idxs],y_test[test_idxs]
            data_t, target_t = data_t.to(device), target_t.flatten().to(device)
            #print(data_t.shape,target_t.shape)
            outputs_t = net(data_t.float())
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/total_test_step)
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    net.train()                                             #read further

#final testing (here validation and test set are switched"
total_v=0
correct_v=0
data_v, target_v=X_validation,y_validation
data_v, target_v = data_v.to(device), target_v.flatten().to(device)
outputs_v = net(data_v.float())
loss_v = criterion(outputs_v, target_v)

_,pred_v = torch.max(outputs_v, dim=1)
correct_v += torch.sum(pred_v==target_v).item()
total_v += target_v.size(0)
val_accu=(100 * correct_v/total_v)
print("The final testing acuracy..............................",val_accu)
fig, ax = plt.subplots()
xs=range(len(train_acc))
ax.plot(xs, train_acc, '--', linewidth=2, label='train')
ax.plot(xs, val_acc, '-', linewidth=2, label='test')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend(loc='lower right')
name="accuracy for Learning Rate:"+str(learning_rate)+".png"
print("savinig plot....", name)
plt.savefig(name)

# In[ ]:



    
    


# In[ ]:





# In[ ]:




