import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# System
import os
import time
import warnings
warnings.filterwarnings("ignore")
# Tools
import argparse
import numpy as np
from tqdm import tqdm
import timm
import random


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help="local gpu id")
parser.add_argument('--world_size', type=int, help="num of processes")

#print(parser.local_rank)
args = parser.parse_args()
world_size=args.world_size
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()
print(global_rank)

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



# THIS IS DATASET PATH AND PARAMS
trainDatapath='/root/commonfile/foodH/train'
valDatapath='/root/commonfile/foodH/test'
BATCH_SIZE = 32
NUM_CLASS = 2173
LR = 0.01


# Load Dataset 
# 2-8 分割
# full_dataset=datasets.ImageFolder(root=trainDatapath,transform=image_transforms['train'])
# train_size=int(len(full_dataset)*0.9)
# val_size=len(full_dataset)-train_size
# train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size])
# 单独的 train test
train_dataset=datasets.ImageFolder(root=trainDatapath,transform=image_transforms['train'])
val_dataset=datasets.ImageFolder(root=valDatapath,transform=image_transforms['val'])


trainsampler = DistributedSampler(train_dataset,rank=args.local_rank)
valsampler = DistributedSampler(val_dataset,rank=args.local_rank)

# train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=trainsampler,num_workers=2,pin_memory=True)
# val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=2,pin_memory=True)
train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=trainsampler,num_workers=4,pin_memory=True)
val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,sampler=valsampler,num_workers=4,pin_memory=True)
#print("train size:",train_size,"; val size:",val_size)

#resnet50 = models.resnet50(pretrained=True)
# 
#resnet50  = timm.create_model('tresnet_m', pretrained=True,num_classes=85)
resnet50  = timm.create_model('tresnet_m_miil_in21k', pretrained=True,num_classes=NUM_CLASS)


resnet50.cuda()
resnet50 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet50)
# Distributed to device
resnet50 = DDP(resnet50, device_ids=[args.local_rank], output_device=args.local_rank)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(),lr=LR,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

def WarmUp(model,optimizer,target_lr,iter):
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
            if(n_iter>ietr):
                break
            

            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            reduce_loss(loss, global_rank, world_size)
            optimizer.step()
    retrun 0



def train_and_valid(model, optimizer, epochs=25):

    # warmup
    WarmUp(model,optimizer,LR,20000)

    history = []
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
 
        train_loss = 0.0
        T_count=0
        V_count=0
        V_k_count=0
        train_acc = 0.0
        #valid_loss = 0.0
        valid_acc = 0.0
        print("This epoch is {} iterations".format(len(train_data)))


        ttime=time.time()
        for i, (inputs, labels) in enumerate(train_data):
            

            inputs = inputs.cuda()
            labels = labels.cuda()
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            # batch*channel*class*prob ???
            

            loss = loss_function(outputs, labels)
            
            loss.backward()
            reduce_loss(loss, global_rank, world_size)
            optimizer.step()

            # New Cal of ACC
            record_gap=20
            if(i%record_gap==record_gap-1):
                ret, predictions = torch.max(outputs, 1)
                correct_counts = torch.eq(predictions, labels).sum().float().item()
                acc1 = correct_counts/inputs.size(0)
                maxk = max((1,3))
                ret,predictions = outputs.topk(maxk,1,True,True)
                predictions = predictions.t()
                correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                acc_topk = correct_counts/inputs.size(0)
                etaTime=(time.time()-ttime)*(len(train_data)-i)/record_gap # not accurate
                loss=loss.mean()
                ETAtime=(1-(i/len(train_data)))*(time.time()-ttime)*(len(train_dataset)/BATCH_SIZE)/record_gap/60
                print("{}/{}\tTop1:{:.2f}%\tTop3:{:.2f}%\tL:{:.5f}\ttime:{:.2f}S\tETA:{:.2f}Min".format(
                    epoch,i,acc1*100,acc_topk*100,loss,time.time()-ttime,ETAtime
                ))
                ttime=time.time()

        
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(val_data):
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs=model(inputs)
                loss = loss_function(outputs, labels)
                loss=loss.mean()

                valid_loss += loss.item() * inputs.size(0)
                # TOP1
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = torch.eq(predictions, labels).sum().float().item()
                V_count += correct_counts
                acc = correct_counts/inputs.size(0)

                # TOP5
                maxk = max((1,3))
                ret,predictions = outputs.topk(maxk,1,True,True)
                predictions = predictions.t()
                correct_counts = predictions.eq(labels.view(1,-1).expand_as(predictions)).sum().item()
                V_k_count += correct_counts
                acc_topk = correct_counts/inputs.size(0)
                #print("Val Loss for {} : {:.5f}\t Top-1 Acc {}%\t Top-3 Acc {}%".format(i,loss,acc*100,acc_topk*100))
                #记录loss
                if(j%100==99):
                    print("Val Loss for {} : {:.5f}\t Top-1 Acc {}%\t Top-3 Acc {}%".format(j,loss,acc*100,acc_topk*100))
                    #writer.add_scalar("LOSS",loss,global_step=i+epoch*len(train_data))
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

        #print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        if not os.path.exists("resnetmodels"):
            os.mkdir("resnetmodels")
        torch.save(model, 'resnetmodels/'+"food"+str(epoch+1)+'.pt')
        #torch.save(metric,'arcmodels/'+"food1500"+'_metric_'+str(epoch+1)+'.pt')
        #writer.add_scalars("acc",{"train_acc":avg_train_acc*100,"val_acc":avg_valid_acc*100,"val_top3_acc":avg_valid_acc_topk*100},global_step=epoch)
        scheduler.step()
        
    return model, history

print("[ START TRAIN ]")
num_epochs = 100
trained_model, history = train_and_valid(resnet50, optimizer, num_epochs)