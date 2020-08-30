import torch
import torch.nn as nn
import time,datetime
# import torch.nn.init as init
# import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
# from utils import *

class Histogram2D(nn.Module):
    def __init__(self, bins=100, min=0, max=1, norm=True, sigma=3, cuda=False):
        super(Histogram2D, self).__init__()
        self.bins, self.min, self.max, self.norm=bins,min,max,norm
        self.sigma=sigma
        self.cuda=cuda
        self.delta=float(max-min)/float(bins)
        self.centers=float(min)+self.delta*(torch.arange(bins).float()+0.5)
        if self.cuda:
            self.centers=self.centers.cuda()
    def forward(self,x):
        # x: tensor(n,h*w,2)
        #making hist using 2 loops
        #cpu, test 1000 times, ~0.20s for data(2*250000*2)
        #gpu, test 1000 times, ~0.05s for data(2*250000*2)
        #gpu, 0.028s for data(2*240*320*2), that is two uv images
        #gpu, 0.072s for data(10*240*320*2), that is two uv images
        counts=torch.zeros(x.shape[0],self.bins, self.bins)
        for i,center1 in enumerate(self.centers):
            for j,center2 in enumerate(self.centers):
                dist=torch.abs(x[:,:,0]-center1)+torch.abs(x[:,:,1]-center2)
                dist=0.5*dist
                ct  =torch.relu(self.delta-dist).sum(1)
                counts[:,i,j]=ct
        if self.norm:
            summ=counts.sum(dim=(1,2))+1e-5
            summ=torch.unsqueeze(summ,1)
            summ=torch.unsqueeze(summ,1)
            return counts/summ
        return counts

class Histogram1D(nn.Module):
    def __init__(self, bins=100, min=0, max=1, norm=True, sigma=1/100, cuda=False):
        super(Histogram1D, self).__init__()
        self.bins, self.min, self.max, self.norm=bins,min,max,norm
        self.sigma=sigma
        self.cuda=cuda
        self.delta=float(max-min)/float(bins)
        self.centers=float(min)+self.delta*(torch.arange(bins).float()+0.5).cuda()
    def forward(self,x):
        #making hist using a for loop, slow
        #cpu, test 1000 times, ~0.0138s for 1 million pixels
        #gpu, test 1000 times, ~0.00398 for 1 million pixels
        counts=[]
        for center in self.centers:
            #dist=F.smooth_l1_loss(x,center, reduction='none')
            dist=torch.abs(x-center)
            ct  =torch.relu(self.delta-dist).sum(1)
            counts.append(ct)
        out = torch.stack(counts,1)
        if self.norm:
            summ=out.sum(1)+1e-5
            return (out.transpose(1,0)/summ).transpose(1,0)
        return out
        #gaussian histogram
        #cpu, test 1000 times, ~0.07 for 1 million pixels
        #gpu, test 1000 times, ~0.007 for 1 million pixels
        # x = x- torch.unsqueeze(self.centers, 1)
        # x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        # x = x.sum(dim=1)
        # x /= x.sum()
        # return x
        return out

if __name__=='__main__':
    print('demo diffihistogram')
    print('test Histogram2D')
    nbatch=2
    data = 64*torch.randn(nbatch*240*320*2).view(nbatch,-1, 2).cuda()
    print(data.shape)
    data.requires_grad=True
    hist2d=Histogram2D(bins=32,min=0,max=64, norm=True, sigma=3, cuda=True)
    out   =hist2d.forward(data)

    #2d plot
    # from matplotlib import pyplot as plt
    # import numpy as np
    # #hist, xedges, yedges = np.histogram2d(x,y)
    # hist=out[1,:,:].cpu().detach().numpy()
    # xedges =hist2d.centers.cpu().numpy()
    # yedges =hist2d.centers.cpu().numpy()
    # X,Y = np.meshgrid(xedges,yedges)
    # plt.imshow(hist)
    # plt.grid(False)
    # plt.colorbar()
    # plt.show()
