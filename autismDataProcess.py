#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import cv2 as cv
from random import shuffle
from tqdm import tqdm
"""
TRAIN_DIR1 = r'\5kImages\train\autistic'
TRAIN_DIR2 = r'\5kImages\train\non_autistic'
VALIDATION_DIR1 = r'\5kImages\valid\autistic'
VALIDATION_DIR2 = r'\5kImages\valid\non_autistic'
TEST_DIR1 = r'\5kImages\test\autistic'
TEST_DIR2 = r'\5kImages\test\non_autistic'
IMG_Resolution = 224
LR = 1e-3
"""




TRAIN_DIR1 = os.path.join('5kImages','train','autistic')
TRAIN_DIR2 = os.path.join('5kImages','train','non_autistic')
VALIDATION_DIR1 = os.path.join('5kImages','valid','autistic')
VALIDATION_DIR2 = os.path.join('5kImages','valid','non_autistic')
TEST_DIR1 =os.path.join('5kImages','test','autistic')
TEST_DIR2 = os.path.join('5kImages','test','non_autistic')
IMG_Resolution = 224
LR = 1e-3

print('trainDirector',TRAIN_DIR1)
print(VALIDATION_DIR1,TEST_DIR1)




def displayImage(img):
    # cv.namedWindow("PALASH", cv.WINDOW_NORMAL)
    # cv.resizeWindow("PALASH", 200, 200)
    cv.imshow("PALASH", img)
    cv.waitKey(10)
    cv.destroyAllWindows()


def create_data(path1, path2):
    dataList=[]
    for imgName in tqdm(os.listdir(path1)):  # first load autistic images
        label = [1]  # [1,0] for autistic part
        fullpath = path1 + '/' + imgName
        # print(fullpath)
        imgData = cv.imread(fullpath, cv.IMREAD_COLOR)
        #print(imgData.shape)
        #displayImage(imgData)
        #print(imgData.shape)
       
        dataList.append([np.array(imgData), np.array(label)])
        #print(dataList[0].shape)
    print(len(dataList))
    for imgName in tqdm(os.listdir(path2)):  # first load autistic images
        label = [0]  # [1,0] for autistic part
        fullpath = path2 + '/' + imgName
        # print(fullpath)
        imgData = cv.imread(fullpath, cv.IMREAD_COLOR)
        # displayImage(imgData)
        dataList.append([np.array(imgData), np.array(label)])
    shuffle(dataList)
   # x = np.array([i[0] for i in dataList]).reshape(-1, IMG_Resolution, IMG_Resolution)
    x = np.array([i[0] for i in dataList])
    x = torch.from_numpy(x)
    x = x.permute(0,3, 1, 2)
    y = np.array([i[1] for i in dataList])
    y=torch.from_numpy(y)
    return x,y


def data_process():
    cwd = os.getcwd()+"/"
    trpath1 = cwd + TRAIN_DIR1  # autistic images
    trpath2 = cwd + TRAIN_DIR2  # regular images
    vpath1 = cwd + VALIDATION_DIR1  # autistic images
    vpath2 = cwd + VALIDATION_DIR2  # regular images
    tstpath1 = cwd + TEST_DIR1  # autistic images
    tstpath2 = cwd + TEST_DIR2  # regular images
    # print(path)

    x_train,y_train=create_data(trpath1, trpath2)
    # np.save('train_data.npy', training_data)

    x_validation,y_validation=create_data(vpath1, vpath2)
    # np.save('validation_data.npy', validation_data)

    x_test,y_test=create_data(tstpath1, tstpath2)
    # np.save('test_data.npy', test_data)
    
    return x_train,y_train,x_validation,y_validation,x_test,y_test


# In[10]:


def main():
    X_train,y_train,X_validation,y_validation,X_test,y_test=data_process()
    print(X_train.shape)
    #X_train = torch.from_numpy(X_train)
    #X_train = X_train.permute(0,3, 1, 2)
    #print(X_train.shape)


# In[11]:


if __name__ == '__main__':    
    main()


# In[ ]:





# In[ ]:




