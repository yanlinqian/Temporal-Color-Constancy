import torch
import torch.nn as nn
import time,datetime
import torch.nn.functional as F
import numpy as np

class Histogram2D_wrap(nn.Module):
    def __init__(self, bins=100, min=0, max=1, norm=True, cuda=False):
        super(Histogram2D_wrap, self).__init__()
        self.bins, self.min, self.max, self.norm=bins,min,max,norm
        self.cuda=cuda
        self.delta=float(max-min)/float(bins)
        self.centers=float(min)+self.delta*(torch.arange(bins).float()+0.5)
        self.eps=1e-5
        if self.cuda:
            self.centers=self.centers.cuda()
    def forward(self,x,mask):
        # x: tensor(n,h*w,2)
        # mask: tensor(n,h*w) where 1 indicates the pixel should be masked, e.g. those on color checker

        x=x-self.min
        x=torch.remainder(x,self.max-self.min)#modular operation
        x=x+self.min
        u, v=x[:,:,0], x[:,:,1]


        #making hist using matrix.
        #speed: for a 240*320 uv image, it takes 0.081s on a 1050ti gpu
        a,b=torch.meshgrid([self.centers, self.centers])
        c  =torch.stack((a.flatten(), b.flatten()))
        c  =c.permute(1,0)#(1024,2)
        x_sparse=x.unsqueeze(2)#([1, 76800, 1, 2])
        c_sparse=c.unsqueeze(0).unsqueeze(0)#([1, 1, 1024, 2])
        x_sparse=x_sparse.expand(-1,-1,self.bins*self.bins,-1)# ([1, 76800, 1024, 2])
        x_sparse=x_sparse.cuda()
        c_sparse=c_sparse.cuda()

        dist=0.5*(torch.abs(x_sparse-c_sparse).sum(3)) # ([1, 76800, 1024])
        ct  =torch.relu(self.delta-dist)# ([1, 76800, 1024])
        ct[torch.isinf(u) | torch.isinf(v) | torch.isnan(u) | torch.isnan(v)]=0
        if mask:
            ct[mask!=0]=0
        ct  =ct.sum(1)
        counts=ct.view(x.shape[0],self.bins, self.bins)


        #making hist using 2 loops, this is pretty slow, for a 240*320 uv image, it takes 0.92s on a 1050ti gpu
        # counts=torch.zeros(x.shape[0],self.bins, self.bins)
        # if self.cuda:
        #     counts=counts.cuda()
        # for i,center1 in enumerate(self.centers):
        #     for j,center2 in enumerate(self.centers):
        #         dist=torch.abs(u-center1)+torch.abs(v-center2)
        #         dist=0.5*dist
        #         ct  =torch.relu(self.delta-dist)
        #         ct[torch.isinf(u)]=0
        #         ct[torch.isinf(v)]=0
        #         ct[torch.isnan(u)]=0
        #         ct[torch.isnan(v)]=0
        #         ct[mask != 0]=0
        #         ct  =ct.sum(1)
        #         counts[:,i,j]=ct


        if self.norm:
            summ=counts.sum(dim=(1,2))+self.eps
            summ=torch.unsqueeze(summ,1)
            summ=torch.unsqueeze(summ,1)
            return counts/summ
        return counts

#not implemented:
#class Histogram1D_wrap

if __name__=='__main__':
    print('demo diffihistogram_wrap')
    nbatch=2
    h,w=64,64
    data = 255*torch.rand(nbatch*h*w*2).view(nbatch,h,w,2).cuda()
    #data[:,100:120,100:120,:]=np.log(0/(0+1e-6))
    data=data.view(nbatch,-1,2)
    mask= (data.sum(2)<=1e-4).float()
    #data = torch.relu(data)
    data.requires_grad=True

    start_time=time.time()
    nrun=10
    hist2d=Histogram2D_wrap(bins=32,min=0,max=64, norm=True,cuda=True)
    for i in range(nrun):
        out   =hist2d.forward(data,mask)
    torch.cuda.synchronize()
    end_time=time.time()-start_time
    print('run time', end_time/nrun)

    print(hist2d.centers, out.shape)

    from showhistogram import *
    x     =hist2d.centers.cpu().numpy()
    y     =x
    z     =out[0,:,:].cpu().detach().numpy()
    show_histogram_2d(x,y,z)
