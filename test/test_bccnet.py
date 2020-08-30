# how to use:
# python ./test/test_bccnet.py --pth_path0 ./trained_models/bccnet/fold0.pth


from __future__ import print_function
import os
import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

sys.path.append('./auxiliary/')
from model import squeezenet1_1,CreateNet
from dataset  import *
from utils import *

sys.path.append('./rcc_net/')
#import rcc-net network
from network import *

#for visualization
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=20)
parser.add_argument('--lrate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--pth_path0', type=str)
parser.add_argument('--pth_path1', type=str)
parser.add_argument('--pth_path2', type=str)
opt = parser.parse_args()

val_loss = AverageMeter()


import scipy.io as scio
#variables to be saved
errors = []
names  = []
preds  = []
gt     = []


#create network
# SqueezeNet = squeezenet1_1(pretrained=True)
# network = CreateNet(SqueezeNet).cuda()
model1 = squeezenet1_1(pretrained=True)
model2 = squeezenet1_1(pretrained=True)
network = CreateNet_rccnet_2squeeze_fcn(model1, model2).cuda()
network.eval()

for i in range(1):
    ############################################test fold 0############################################
    #dataset_test = ColorChecker(train=False,folds_num=i)
    dataset_test = Temporal(train=False, onlylastimg=False, input_size=[224,224], mimic=True)


    #dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,shuffle=False, num_workers=opt.workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)
    len_dataset_test = len(dataset_test)
    print('Len_fold:',len(dataset_test))
    if i == 0:
        pth_path = opt.pth_path0

    #load parameters
    if os.path.exists(pth_path):
        print('found parameters', pth_path)
        network.load_state_dict(torch.load(pth_path))
    network.eval()
    with torch.no_grad():
        for i,data in enumerate(dataloader_test):
            img, img_mimic, label,fn = data
            #print('img statistic:', img.shape, img )
            img = Variable(img.cuda())
            img_mimic = Variable(img_mimic.cuda())
            label = Variable(label.cuda())
            pred = network(img, img_mimic)
            #print('pred.shape', pred.shape)#(1,3,23,35)
            #mean of patches
            pred_ill = torch.nn.functional.normalize(torch.sum(torch.sum(pred,2),2),dim=1)
            # #median of patches
            # pred_r, pred_g, pred_b=pred[:,0,:,:],pred[:,1,:,:],pred[:,2,:,:]
            # pred_r=torch.median(torch.flatten(pred_r)).view(1,-1)
            # pred_g=torch.median(torch.flatten(pred_g)).view(1,-1)
            # pred_b=torch.median(torch.flatten(pred_b)).view(1,-1)
            # pred_ill=torch.cat((pred_r, pred_g, pred_b),dim=1)

            loss = get_angular_loss(pred_ill,label)
            val_loss.update(loss.item())

            errors.append(loss.item())
            names.append(fn[0])
            preds.append(pred_ill.cpu().numpy())
            gt.append(label.cpu().numpy())

            if False: #draw figure (pipeline of tccnet) for our paper
                print('img',img.shape)
                print('img_mimic',img_mimic.shape)
                print('pred', pred.shape)
                # img torch.Size([1, 8, 3, 224, 224])
                # img_mimic torch.Size([1, 8, 3, 224, 224])
                # pred torch.Size([1, 3, 7, 7])


                print('corrected')
                nonlinear_img=img[0]
                corrected_img=correct_image_linear(nonlinear_img, label)
                img_show=corrected_img[-1,:,:,:].permute(1,2,0)
                #img_show=torch.clamp(img_show,0,1)
                img_show=img_show/img_show.max()
                img_show=torch.clamp(img_show,0,1)
                plt.imshow(img_show.detach().cpu().numpy());
                plt.axis('off')
                plt.show()

                print('pred')
                img_show=pred[0,:,:,:].permute(1,2,0)
                plt.imshow(img_show.detach().cpu().numpy());
                plt.axis('off')
                plt.show()


                print('real img')
                for j in range(img.shape[1]-8,img.shape[1],2):
                    img_show=img[0,j,:,:,:].permute(1,2,0)#feature[j,:,:]
                    img_show=img_show
                    plt.imshow(img_show.detach().cpu().numpy());
                    plt.axis('off')
                    plt.show()
                print('fake img')
                for j in range(img.shape[1]-8,img.shape[1],2):
                    img_show=img_mimic[0,j,:,:,:].permute(1,2,0)#feature[j,:,:]
                    img_show=img_show
                    plt.imshow(img_show.detach().cpu().numpy());
                    plt.axis('off')
                    plt.show()

            if i%10==0:
                print('Model: %s, AE: %f'%(fn[0],loss.item()))
            scio.savemat('%s/%s.mat' % ('results','tccnet_results'), {'errors':errors, 'names':names, 'preds':preds, 'gt':gt})


mean,median,trimean,bst25,wst25,pct95 = evaluate(errors)
print('Mean: %f, Med: %f, tri: %f, bst: %f, wst: %f, pct: %f'%(mean,median,trimean,bst25,wst25,pct95))
