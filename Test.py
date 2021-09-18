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
model=timm.create_model('tresnet_m_miil_in21k', pretrained=True,num_classes=10)
print(model.head.fc)
#Linear(in_features=2048, out_features=10, bias=True)
# ArcFaceNet-> input:features   out:loss
class NewFC(nn.Module):
    # 返回 features 和 out 的FC层，便于计算损失
    def __init__(self,in_features,out_features):
        super(NewFC,self).__init__()
        self.fc=nn.Linear(in_features=in_features,out_features=out_features)
    def forward(self,features):
        out=self.fc(features)
        return features,out

class ArcFaceNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=2):
        super(ArcFaceNet, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, cls_num))

    def forward(self, features, m=1, s=10):
        # 特征与权重 归一化
        _features = nn.functional.normalize(features, dim=1)
        _w = nn.functional.normalize(self.w, dim=0)
        # 特征向量与参数向量的夹角theta，分子numerator，分母denominator
        theta = torch.acos(torch.matmul(_features, _w) / 10)  # /10防止下溢
        numerator = torch.exp(s * torch.cos(theta + m))
        denominator = torch.sum(torch.exp(s * torch.cos(theta)), dim=1, keepdim=True) - torch.exp(
            s * torch.cos(theta)) + numerator
        return torch.log(torch.div(numerator, denominator))
print(model)
model.head.fc=NewFC(2048,1000)
CELoss= nn.CrossEntropyLoss()
ARCLoss=ArcFaceNet(1000,2048)

optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
features,outputs=model(images)
print(features.shape)
print(outputs.shape)
ce_loss=CELoss(outputs,labels)
arc_loss=ARCLoss(features,labels)



# optimizer = optim.SGD(resnet50.parameters(),lr=0.01,momentum=0.9)
# print(optimizer)
# optimizer.param_groups[0]['lr']=1
# print(optimizer)

# for i in range(100):
#     time.sleep(0.1)
#     print('\r', i, end='', flush=True)
# image_transforms = {
#     'train':transforms.Compose([
# 	    #transforms.ToPILImage(),
#         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomHorizontalFlip(),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                             [0.229, 0.224, 0.225])
#     ]),
#     'val':transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                             [0.229, 0.224, 0.225])
#     ])
# }

# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, folder, klass, transform):
#         self._data = folder
#         self.klass = klass
#         self.transform=transform
#         self.imgs=glob.glob(self._data+'/*')
#         #self.extension = extension
#         # Only calculate once how many files are in this folder
#         # Could be passed as argument if you precalculate it somehow
#         # e.g. ls | wc -l on Linux
#         self._length = sum(1 for entry in os.listdir(self._data))

#     def __len__(self):
#         # No need to recalculate this value every time
#         return self._length

#     def __getitem__(self, index):
#         # images always follow [0, n-1], so you access them directly
#         img=Image.open(self.imgs[index])
#         img=transform(img)
#         return img,self.kclass

