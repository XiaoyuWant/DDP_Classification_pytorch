### For TEST USE
### Not 
### For 
### Train
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import timm
import time
import os
import glob

# resnet50=timm.create_model('tresnet_m_miil_in21k', pretrained=True,num_classes=10)
# optimizer = optim.SGD(resnet50.parameters(),lr=0.01,momentum=0.9)
# print(optimizer)
# optimizer.param_groups[0]['lr']=1
# print(optimizer)

# for i in range(100):
#     time.sleep(0.1)
#     print('\r', i, end='', flush=True)

