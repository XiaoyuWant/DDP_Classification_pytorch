from urllib import parse
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import Parameter
import torch.nn.functional as F
# System
import os
import time
import warnings
import glob
warnings.filterwarnings("ignore")
# Tools
import math
import argparse
import numpy as np
from tqdm import tqdm
import timm
import random
from PIL import Image
# test
# !pip install from prefetch_generator import BackgroundGenerator
# from prefetch_generator import BackgroundGenerator
# from torch.cuda.amp import autocast as autocast


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")
parser.add_argument('--batchsize',type=int,defalut=32)

#print(parser.local_rank)
args = parser.parse_args()
world_size=args.world_size
# dist.init_process_group(backend='nccl', init_method='env://')
dist.init_process_group(backend='nccl',init_method='env://',rank=args.local_rank,world_size=args.world_size)
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()
print(global_rank)

torch.backends.cudnn.benchmark=True
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('food/runs/exp2')
def set_seed(seed):
    #必须禁用模型初始化中的任何随机性。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.set_deterministic(True)
set_seed(999)

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size

image_transforms = {
    'train':transforms.Compose([
	    #transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(size=224),
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



# THIS IS DATASET PATH AND PARAMS
trainDatapath='/root/commonfile/foodH/train'
valDatapath='/root/commonfile/foodH/test'

BATCH_SIZE = 4*2
NUM_CLASS = 2173
LR = 0.001
NUM_EPOCH = 100


# Load Dataset 
# 2-8 分割
# full_dataset=datasets.ImageFolder(root=trainDatapath,transform=image_transforms['train'])
# train_size=int(len(full_dataset)*0.9)
# val_size=len(full_dataset)-train_size
# train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size])
# 单独的 train test

# For Large Dataset Need Write a new Dataloader because torchvision.ImageFolder has a problem

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())



root="/root/commonfile/foodH/"

class NewFC(nn.Module):
    # 返回 features 和 out 的FC层，便于计算损失
    def __init__(self,in_features,out_features):
        super(NewFC,self).__init__()
        self.fc=nn.Linear(in_features=in_features,out_features=out_features)
    def forward(self,features):
        features=self.fc(features)
        return features

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
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # 加上一个softmax
        # arc_outputs 是未经softmax层的
        # self.softmax =nn.Softmax(dim=0)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        #加上一个softmax
        # output = self.softmax(output)
        return output


train_dataset=datasets.ImageFolder(root=trainDatapath,transform=image_transforms['train'])
val_dataset=datasets.ImageFolder(root=valDatapath,transform=image_transforms['val'])


trainsampler = DistributedSampler(train_dataset,rank=args.local_rank)
valsampler = DistributedSampler(val_dataset,rank=args.local_rank)

# train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=trainsampler,num_workers=2,pin_memory=True)
# val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=2,pin_memory=True)
train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=trainsampler,num_workers=2,pin_memory=True)
val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=2,pin_memory=True)
#print("Train size:",train_size,"; val size:",val_size)

#resnet50 = models.resnet50(pretrained=True)
#resnet50  = timm.create_model('tresnet_m', pretrained=True,num_classes=85)
resnet50  = timm.create_model('tresnet_m_miil_in21k', pretrained=True,num_classes=NUM_CLASS)
resnet50.head.fc=NewFC(2048,256)
ARC=ArcMarginProduct(256,NUM_CLASS, s=30, m=0.5, easy_margin=False)


# Distributed to device
resnet50.cuda()
resnet50 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet50)
resnet50 = DDP(resnet50, device_ids=[args.local_rank], output_device=args.local_rank)
ARC.cuda()
ARC=torch.nn.SyncBatchNorm.convert_sync_batchnorm(ARC)
ARC=DDP(ARC, device_ids=[args.local_rank], output_device=args.local_rank)
softmax=nn.LogSoftmax(dim=0).cuda()
loss_function = nn.CrossEntropyLoss().cuda()
# CrossEntropy = LogSoftmax + NLLLoss
#loss_function = nn.NLLLoss().cuda()
optimizer = torch.optim.SGD([{'params': resnet50.parameters()}, {'params': ARC.parameters()}],
                                    lr=LR, weight_decay=5e-4,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def WarmUp(model,optimizer,target_lr,iter):
    if(args.local_rank==0):
        print("Warm up for iterations of:",str(iter))
    model.train()
    begin_lr=1e-6
    n_iter=0
    while(n_iter < iter):
        for i, (inputs, labels) in enumerate(train_data):
            # Set learning rate by iter
            # and update lr to warm up learning

            lr=begin_lr+n_iter*(target_lr-begin_lr)/iter
            optimizer.param_groups[0]['lr']=lr
            n_iter += 1
            if(args.local_rank==0):
                info="iter:\t{}\tlr:\t{:.5f}".format(n_iter,lr)
                print('\r', info, end='', flush=True)
            if(n_iter>iter):
                break
            

            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            reduce_loss(loss, global_rank, world_size)
            optimizer.step()
    return 0



def train_and_valid(model, optimizer, epochs=25):

    # warmup
    #WarmUp(model,optimizer,LR,20000)

    history = []
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        #epoch_start = time.time()
        if(args.local_rank==0):
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print("This epoch is {} iterations".format(len(train_data)))
        model.train()
        ARC.train()
 
        ttime=time.time()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            features= model(inputs)
            arc_outputs = ARC(features,labels)
            arc_loss=loss_function(arc_outputs,labels)
            loss=arc_loss
            torch.distributed.barrier()
            loss.backward()
            optimizer.step()
            

            # New Cal of ACC
            record_gap=20
            if(i%record_gap==record_gap-1 and args.local_rank==0):
                #print(arc_outputs)
                arc_outputs=softmax(arc_outputs)
                ret, predictions = torch.max(arc_outputs, 1)
                correct_counts = torch.eq(predictions, labels).sum().float().item()
                acc1 = correct_counts/inputs.size(0)
                maxk = max((1,3))
                ret,predictions = arc_outputs.topk(maxk,1,True,True)
                predictions = predictions.t()
                correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                acc_topk = correct_counts/inputs.size(0)
                etaTime=(time.time()-ttime)*(len(train_data)-i)/record_gap # not accurate
                loss=loss.mean()
                ETAtime=(1-(i/len(train_data)))*(time.time()-ttime)*(len(train_dataset)/BATCH_SIZE)/record_gap/60
                info="{}/{}\tTop1:{:.2f}%\tTop3:{:.2f}%\tL:{:.5f}\tarc:{:.4f}\ttime:{:.2f}S\tETA:{:.2f}Min".format(
                    epoch,i,acc1*100,acc_topk*100,loss,arc_loss, time.time()-ttime,ETAtime
                )
                print('\r',info,end=' ',flush=True)
                ttime=time.time()

        
            valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            ARC.eval()
            T_count=0
            V_count=0
            V_k_count=0
            for j, (inputs, labels) in enumerate(val_data):
                inputs = inputs.cuda()
                labels = labels.cuda()

                features=model(inputs)
                arc_outputs = ARC(features)
                arc_loss=loss_function(arc_outputs,labels)
                loss=arc_loss
                loss=loss.mean()


                arc_outputs=softmax(arc_outputs)


                valid_loss += loss.item() * inputs.size(0)
                # TOP1
                ret, predictions = torch.max(arc_outputs.data, 1)
                correct_counts = torch.eq(predictions, labels).sum().float().item()
                V_count += correct_counts
                acc = correct_counts/inputs.size(0)

                # TOP5
                maxk = max((1,3))
                ret,predictions = arc_outputs.topk(maxk,1,True,True)
                predictions = predictions.t()
                correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                V_k_count += correct_counts
                acc_topk = correct_counts/inputs.size(0)

                #记录loss
                if(j%100==99):
                    print("Val Loss for {} : {:.5f}\t Top-1 Acc {}%\t Top-3 Acc {}%".format(j,loss,acc*100,acc_topk*100))
            loss_of_val=valid_loss*world_size/len(val_dataset)
            top1_of_val=V_count*world_size/len(val_dataset)
            top3_of_val=V_k_count*world_size/len(val_dataset)
            print("VAL:{}\tTop1:{:.2f}%\tTop3:{:.2f}%\tL:{:.5f}".format(
                        epoch,top1_of_val*100,top3_of_val*100,loss_of_val
                    ))


            # 保存记录
            with open("output.txt",'a') as f:
                text="Epoch{}\tTop1:\t{}\tTop3:\t{}\n".format(epoch,top1_of_val*100,top3_of_val*100 )
                f.write(text)


        if not os.path.exists("resnetmodels"):
            os.mkdir("resnetmodels")
        torch.save(model, 'resnetmodels/'+"food"+str(epoch+1)+'.pt')
        #torch.save(metric,'arcmodels/'+"food1500"+'_metric_'+str(epoch+1)+'.pt')
        #writer.add_scalars("acc",{"train_acc":avg_train_acc*100,"val_acc":avg_valid_acc*100,"val_top3_acc":avg_valid_acc_topk*100},global_step=epoch)
        scheduler.step()
        
    return model, history


trained_model, history = train_and_valid(resnet50, optimizer, NUM_EPOCH)