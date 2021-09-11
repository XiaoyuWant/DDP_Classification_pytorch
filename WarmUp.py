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
from torch.utils.data import ConcatDataset
# resnet50=timm.create_model('tresnet_m_miil_in21k', pretrained=True,num_classes=10)
# optimizer = optim.SGD(resnet50.parameters(),lr=0.01,momentum=0.9)
# print(optimizer)
# optimizer.param_groups[0]['lr']=1
# print(optimizer)

# for i in range(100):
#     time.sleep(0.1)
#     print('\r', i, end='', flush=True)
image_transforms = {
    'train':transforms.Compose([
	    #transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
}

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, klass, transform):
        self._data = folder
        self.klass = klass
        self.transform=transform
        self.imgs=glob.glob(self._data+'/*')
        #self.extension = extension
        # Only calculate once how many files are in this folder
        # Could be passed as argument if you precalculate it somehow
        # e.g. ls | wc -l on Linux
        self._length = sum(1 for entry in os.listdir(self._data))

    def __len__(self):
        # No need to recalculate this value every time
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        img=Image.open(self.imgs[index])
        img=transform(img)
        return img,self.kclass

root="/root/commonfile/foodH/"

folders=glob.glob(root+"test/*")
index=0
val_dataset=[]
for folder in folders:
    val_dataset.append(ImageDataset(folder,index,image_transforms['val']))
    index+=1
val_dataset=ConcatDataset(val_dataset)
print(len(val_dataset))
