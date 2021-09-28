import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import json
import os
import argparse

from torchvision import datasets

import utils 
from model import model 

import itertools
import numpy as np 

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


### ------------------------------------ Dataloader -------------------------------------- ### 
def get_dataloader(dataset, train_dir, val_dir, batchsize): 
    
    if 'CIFAR' in dataset : 
        if dataset == 'CIFAR10': 
            norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            nb_cls = 10
            
        elif dataset == 'CIFAR100':
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            nb_cls = 100
        elif dataset== 'food85':
            norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
        # transformation of the training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        # transformation of the validation set
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])
            
    elif dataset == 'Clothing1M':
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        nb_cls = 2173

        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

    # full_dataset=ImageFolder(train_dir, transform_train)
    # train_size=int(len(full_dataset)*0.85)
    # val_size=len(full_dataset)-train_size
    # trainset,valset=torch.utils.data.random_split(full_dataset,[train_size,val_size])
    trainset=ImageFolder(train_dir,transform_train)
    valset=ImageFolder(val_dir,transform_val)
    trainloader = DataLoader(trainset, 
                             batch_size=batchsize, 
                             shuffle=True, 
                             drop_last=True,
                             num_workers=8,
                             pin_memory=True)

    valloader = DataLoader(valset,
                           batch_size=batchsize, 
                           shuffle=False, 
                           drop_last=False, 
                           num_workers=8, 
                           pin_memory=True)   
                                
    return  trainloader, valloader, nb_cls  

### -------------------------------------------------------------------------------------------- 


### ------------------------------------ Distribution -------------------------------------- ###  
def GaussianDist(mu, std, N):
    
    dist = np.array([np.exp(-((i - mu) / std)**2) for i in range(1, N + 1)])
    
    return dist / np.sum(dist)

### --------------------------------------------------------------------------------------------
    

### ------------------------ Test with Nested (iterate all possible K) --------------------- ### 
def TestNested(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim, dropout): 
    
    net_feat.eval() 
    net_cls.eval()
    
    bestTop1 = 0
    
    true_pred = torch.zeros(len(mask_feat_dim)).cuda()
    top3_pred=torch.zeros(len(mask_feat_dim)).cuda()
    nb_sample = 0    
    top3_correct_count=0

    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
             
        feature = net_feat(inputs)
        outputs = []

        for i in range(len(mask_feat_dim)) : 
            feature_mask = feature * mask_feat_dim[i]
            outputs.append(net_cls(feature_mask).unsqueeze(0)) 
        
        outputs = torch.cat(outputs, dim=0)
        # print(outputs.shape)
        # print(targets.shape)
        
        _, pred = torch.max(outputs, dim=2)
        targets = targets.unsqueeze(0).expand_as(pred)
        
        true_pred = true_pred + torch.sum(pred == targets, dim=1).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

        # top3 
        maxk=max((1,3))
        ret,predictions = outputs.topk(maxk,2,True,True)
        targets=targets.unsqueeze(2).expand_as(predictions)
        top3_correct_count =top3_correct_count +torch.sum(predictions == targets, dim=1).type(torch.cuda.FloatTensor)
        
        
    acc, k = torch.max((true_pred / nb_sample - 1e-5 * torch.arange(len(mask_feat_dim)).type_as(true_pred)), dim=0)
    acc, k = acc.item(), k.item()
    acc3=top3_correct_count/nb_sample
    acc3=acc3[k]
    acc3=torch.sum(acc3,dim=0).item()
    
    msg = '\nNested ... Epoch {:d}, Acc {:.3f} %, K {:d} (Best Acc {:.3f} %)'.format(epoch, acc * 100, k, best_acc * 100)
    print (msg)
    print("ACC\tTop1:",acc,'\tTop3:',acc3)
    
    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc, acc)
        print (msg)
        print ('Saving Best!!!')
        param = {'feat': net_feat.state_dict(), 
                 'cls': net_cls.state_dict(), 
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))
        
        best_acc = acc
        best_k = k
        
    return best_acc, acc,acc3, best_k

### --------------------------------------------------------------------------------------------
    

### ---------------- Test standard (used for model w/o nested, baseline, dropout) ---------- ###   
def TestStandard(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim, dropout):
    net_feat.eval() 
    net_cls.eval()
    bestTop1 = 0
    true_pred = torch.zeros(1).cuda()
    nb_sample = 0
    top3_correct_count=0
    for batchIdx, (inputs, targets) in enumerate(valloader):
        inputs = inputs.cuda() 
        targets = targets.cuda()
        
        feature = net_feat(inputs)
        feature = F.dropout(feature, p=dropout, training=False)
        outputs = net_cls(feature)
        
        _, pred = torch.max(outputs, dim=1)
        
        true_pred = true_pred + torch.sum(pred == targets).type(torch.cuda.FloatTensor)
        nb_sample += len(inputs)

        ## top3
        maxk=max((1,3))
        ret,predictions = outputs.topk(maxk,1,True,True)
        predictions = predictions.t()
        correct_counts = predictions.eq(targets.view(1,-1).expand_as(predictions)).sum().item()
        top3_correct_count+=correct_counts


    acc = true_pred / nb_sample
    acc = acc.item()
    acc3=top3_correct_count/nb_sample
    print("ACC\tTop1:",acc,'\tTop3:',acc3)

    
    # msg = 'Standard ... Epoch {:d}, Acc {:.3f} %, (Best Acc {:.3f} %)'.format(epoch, acc * 100, best_acc * 100)
    # print (msg)
    
    # save checkpoint
    if acc > best_acc:
        msg = 'Best Performance improved from {:.3f} --> {:.3f}'.format(best_acc * 100, acc * 100)
        print (msg)
        print ('Saving Best!!!')
        param = {'feat': net_feat.state_dict(), 
                 'cls': net_cls.state_dict(), 
                 }
        torch.save(param, os.path.join(out_dir, 'netBest.pth'))
        
        best_acc = acc
        
    return best_acc, acc,acc3, len(mask_feat_dim)

### --------------------------------------------------------------------------------------------


### -------------------------------------- Training  --------------------------------------- ###  
def Train(epoch, optimizer, net_feat, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn):   
    
    msg = '\nEpoch: {:d}'.format(epoch)
    print (msg)
    net_feat.train(freeze_bn = freeze_bn)
    net_cls.train()
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()    
        
    for batchIdx, (inputs, targets) in enumerate(trainloader):   
        inputs = inputs.cuda() 
        targets = targets.cuda()
        
        for optim in optimizer: 
            optim.zero_grad()

        feature = net_feat(inputs)

        if dist is not None: 
            k = np.random.choice(range(len(mask_feat_dim)), p=dist)
            mask_k = mask_feat_dim[k]
            feature_masked = feature * mask_k
        else:
            feature_masked = F.dropout(feature, p=dropout, training=True)

        outputs = net_cls(feature_masked)
            
        loss = criterion(outputs, targets)

        loss.backward()
        for optim in optimizer: 
            optim.step()
            
        acc1, acc3 = utils.accuracy(outputs, targets, topk=(1, 3))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(acc1[0].item(), inputs.size()[0])
        top3.update(acc3[0].item(), inputs.size()[0])

        msg = 'Loss: {:.3f} | Top1: {:.3f}% | Top3: {:.3f}%'.format(losses.avg, top1.avg, top3.avg)
        utils.progress_bar(batchIdx, len(trainloader), msg)
        
    return losses.avg, top1.avg, top3.avg

### --------------------------------------------------------------------------------------------   


### ------------------------------------ Lr Warm Up  --------------------------------------- ###   
def LrWarmUp(warmUpIter, lr, optimizer, net_feat, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn) : 
    
    nbIter = 0 
    
    while nbIter < warmUpIter: 
        net_feat.train(freeze_bn = freeze_bn)
        net_cls.train()
        
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top3 = utils.AverageMeter()    
        
        for batchIdx, (inputs, targets) in enumerate(trainloader):
            nbIter += 1
            if nbIter == warmUpIter: 
                break
            lrUpdate = nbIter / float(warmUpIter) * lr
            for optim in optimizer: 
                for g in optim.param_groups:
                    g['lr'] = lrUpdate
                
            inputs = inputs.cuda() 
            targets = targets.cuda() 
            
            for optim in optimizer: 
                optim.zero_grad()

            feature = net_feat(inputs)

            if dist is not None: 
                k = np.random.choice(range(len(mask_feat_dim)), p=dist)
                mask_k = mask_feat_dim[k]
                feature_masked = feature * mask_k
            else:
                feature_masked = F.dropout(feature, p=dropout, training=True) 

            outputs = net_cls(feature_masked) 
                
            loss = criterion(outputs, targets)

            loss.backward()
            for optim in optimizer: 
                optim.step()
                
            acc1, acc3 = utils.accuracy(outputs, targets, topk=(1, 3))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            top3.update(acc3[0].item(), inputs.size()[0])

            msg = 'Loss: {:.3f} | Lr : {:.5f} | Top1: {:.3f}% | Top3: {:.3f}%'.format(losses.avg, lrUpdate, top1.avg, top3.avg)
            utils.progress_bar(batchIdx, len(trainloader), msg)

### -------------------------------------------------------------------------------------------- 
  
            
#-----------------------------------------------------------------------------------------------                       
#-----------------------------------------------------------------------------------------------            
########################################-- MAIN FUNCTION --#####################################
#-----------------------------------------------------------------------------------------------                         
#-----------------------------------------------------------------------------------------------          
          
def main(gpu, arch, dropout, out_dir, dataset, train_dir, val_dir, warmUpIter, lr, nbEpoch, batchsize, momentum=0.9, weightDecay = 5e-4, lrSchedule = [200, 300], mu=0, nested=1.0, resumePth=None, freeze_bn=False, pretrained=True): 

    best_acc = 0  # best test accuracy
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    trainloader, valloader, nb_cls = get_dataloader(dataset, train_dir, val_dir, batchsize)
    
    # feature net + classifier net (a linear layer)
    net_feat = model.NetFeat(arch = arch,
                             pretrained = pretrained,
                             dataset = dataset)
                       
    net_cls = model.NetClassifier(feat_dim = net_feat.feat_dim,
                                  nb_cls = nb_cls)
    
    net_feat.cuda()
    net_cls.cuda()
    feat_dim = net_feat.feat_dim
    best_k = feat_dim

    # generate mask 
    mask_feat_dim = []
    for i in range(feat_dim): 
        tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
        tmp[:, : (i + 1)] = 1
        mask_feat_dim.append(tmp)

    # distribution and test function
    # 设置了Nested Dropout的区别就在于这里？

    dist = GaussianDist(mu, nested, feat_dim) if nested > 0 else None

    Test = TestNested if nested > 0 else TestStandard

    # load model
    if resumePth : 
        param = torch.load(resumePth)
        net_feat.load_state_dict(param['feat'])
        print ('Loading feature weight from {}'.format(resumePth))
        
        net_cls.load_state_dict(param['cls'])
        print ('Loading classifier weight from {}'.format(resumePth))
        
    
    # output dir + loss + optimizer    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = [torch.optim.SGD(itertools.chain(*[net_feat.parameters()]), 
                                                1e-4, 
                                                momentum=args.momentum),
                                                
                torch.optim.SGD(itertools.chain(*[net_cls.parameters()]), 
                                                1e-4, 
                                                momentum=args.momentum)] # remove the weight decay in classifier
    # optimizer = [torch.optim.Adam(itertools.chain(*[net_feat.parameters()]), 
    #                                             1e-4),
    #             torch.optim.Adam(itertools.chain(*[net_cls.parameters()]), 
    #                                             1e-4)]

    # optimizer = [torch.optim.SGD(itertools.chain(*[net_feat.parameters()]), 
    #                                              1e-7, 
    #                                              momentum=args.momentum, 
    #                                              weight_decay=args.weightDecay),
                                                 
    #              torch.optim.SGD(itertools.chain(*[net_cls.parameters()]), 
    #                                              1e-7, 
    #                                              momentum=args.momentum, 
    #                                              weight_decay=args.weightDecay)] # remove the weight decay in classifier
                
    # learning rate warm up
    print("Start warmUp")
    LrWarmUp(warmUpIter, lr, optimizer, net_feat, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn)
    print("End warmUp")

    with torch.no_grad(): 
        best_acc, acc,acc3, best_k = Test(0, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim, dropout)
        
    best_acc, best_k = 0, feat_dim
    for optim in optimizer: 
        for g in optim.param_groups:
            g['lr'] = lr
           
    history = {'trainTop1':[], 'best_acc':[], 'trainTop3':[], 'valTop1':[],'valTop3':[], 'trainLoss':[], 'best_k':[]}

    lrScheduler = [MultiStepLR(optim, milestones=lrSchedule, gamma=0.1) for optim in optimizer]
    
    for epoch in range(nbEpoch):  
        trainLoss, trainTop1, trainTop3 = Train(epoch, optimizer, net_feat, net_cls, trainloader, criterion, dist, mask_feat_dim, dropout, freeze_bn)
        with torch.no_grad() : 
            best_acc, valTop1, valTop3,best_k = Test(epoch, best_acc, best_k, net_feat, net_cls, valloader, out_dir, mask_feat_dim, dropout)
            # save to output.txt
            with open("output.txt",'a') as f:
                text="【{}】\tTop1:\t{}\tTop3:\t{}\n".format(epoch,valTop1,valTop3)
                f.write(text)

            
        history['trainTop1'].append(trainTop1)
        history['trainTop3'].append(trainTop3)
        history['trainLoss'].append(trainLoss)
        history['valTop1'].append(valTop1)
        history['valTop3'].append(valTop3)
        
        history['best_acc'].append(best_acc)
        history['best_k'].append(best_k)
        
        with open(os.path.join(out_dir, 'history.json'), 'w') as f : 
            json.dump(history, f)
        
        for lr_schedule in lrScheduler: 
            lr_schedule.step()

    msg = 'mv {} {}'.format(out_dir, '{}_Acc{:.3f}_K{:d}'.format(out_dir, best_acc, best_k))
    print (msg)
    os.system(msg)
    
    

if __name__ == '__main__': 
                        
    parser = argparse.ArgumentParser(description='PyTorch Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    parser.add_argument('--train-dir', type=str, default='../foodH/train', help='train directory')
    parser.add_argument('--val-dir', type=str, default='../foodH/test', help='val directory')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100', 'Clothing1M'], default='Clothing1M', help='which dataset?')   
    
    # training
    parser.add_argument('--warmUpIter', type=int, default=15000, help='total iterations for learning rate warm')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--nbEpoch', type=int, default=150, help='nb epoch')
    parser.add_argument('--lrSchedule', nargs='+', type=int, default=[20, 30,40,120], help='lr schedule') 
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

    # model
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet50', help='which archtecture?')
    parser.add_argument('--out-dir', type=str,default='./out/' ,help='output directory')
    parser.add_argument('--mu', type=float, default=0.0, help='nested mean hyperparameter')
    parser.add_argument('--nested', type=float, default=0.0, help='nested std hyperparameter')  
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')  
    parser.add_argument('--resumePth', type=str, help='resume path')
    parser.add_argument('--freeze-bn', action='store_true', help='freeze the BN layers')  
    #parser.add_argument('--freeze-bn', default=False, help='freeze the BN layers') 
    parser.add_argument('--pretrained', action='store_true', help='Start with ImageNet pretrained model (Pytorch Model Zoo)')
    
    args = parser.parse_args()
    print (args)
    
    if args.nested > 0 and args.dropout > 0 : 
        raise RuntimeError('Activating both nested (eta = {:.3f}) and dropout (ratio = {:.3f})'.format(args.nested, args.dropout))
    # BASELINE    
    # main(gpu = args.gpu, 
    #      arch = args.arch,          
    #      dropout = args.dropout,         
    #      out_dir = args.out_dir,          
    #      dataset = args.dataset,          
    #      train_dir = args.train_dir,          
    #      val_dir = args.val_dir,          
    #      warmUpIter = args.warmUpIter,         
    #      lr = args.lr,         
    #      nbEpoch = args.nbEpoch,       
    #      batchsize = args.batchsize,        
    #      momentum = args.momentum,         
    #      weightDecay = args.weightDecay,         
    #      lrSchedule = args.lrSchedule,
    #      mu = args.mu,         
    #      nested = args.nested,         
    #      resumePth = args.resumePth,
    #      freeze_bn = args.freeze_bn,        
    #      pretrained = True)
    # NESTED
    main(gpu = args.gpu, 
         arch = args.arch,          
         dropout = args.dropout,         
         out_dir = args.out_dir,          
         dataset = args.dataset,          
         train_dir = args.train_dir,          
         val_dir = args.val_dir,          
         warmUpIter = args.warmUpIter,         
         lr = args.lr,         
         nbEpoch = args.nbEpoch,       
         batchsize = args.batchsize,        
         momentum = args.momentum,         
         weightDecay = args.weightDecay,         
         lrSchedule = args.lrSchedule,
         mu = args.mu,         
         nested = 100,         
         resumePth = args.resumePth,
         freeze_bn = True,        
         pretrained = True)
