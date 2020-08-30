import math
import cv2
import torch
import numpy as np
from math import *

def rgb_to_uv(rgb):
    #rgb: tensor (n,h,w,3)
    assert len(rgb.shape)==4 or len(rgb.shape)==2
    eps=1e-6
    if len(rgb.shape)==4:
        uv=torch.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2],2))
        uv[:,:,:,0]=torch.log(rgb[:,:,:,1]/(rgb[:,:,:,0]+eps))
        uv[:,:,:,1]=torch.log(rgb[:,:,:,1]/(rgb[:,:,:,2]+eps))
    if len(rgb.shape)==2:
        uv=torch.zeros((rgb.shape[0], 2))
        uv[:,0]=torch.log(rgb[:,1]/(rgb[:,0]+eps))
        uv[:,1]=torch.log(rgb[:,1]/(rgb[:,2]+eps))
    return uv

def uv_to_rgb(uv):
    #uv: tensor(n,h,w,2) OR (n,2)
    assert len(uv.shape)==4 or len(uv.shape)==2
    if len(uv.shape)==4:
        r=torch.exp(-uv[:,:,:,0])
        g=torch.ones(uv.shape[0], uv.shape[1], uv.shape[2]).dobule().cuda()
        b=torch.exp(-uv[:,:,:,1])
        rgb=torch.stack((r,g,b), dim=3)
        rgb=rgb/ torch.sqrt((rgb**2).sum(3, keepdim=True))
        return rgb
    if len(uv.shape)==2:
        r=torch.exp(-uv[:,0])
        g=torch.ones(uv.shape[0]).double().cuda()
        b=torch.exp(-uv[:,1])
        rgb=torch.stack((r,g,b), dim=1)
        rgb=rgb/ torch.sqrt((rgb**2).sum(1, keepdim=True))
        return rgb

if __name__ == '__main__':
    print('demo rgbuv')

    file_path='./auxiliary/8D5U5524.png'
    raw=np.array(cv2.imread(file_path,-1), dtype='float32')
    raw=np.maximum(raw-1,[0,0,0])
    img= (np.clip(raw / raw.max(), 0, 1) * 1.0).astype(np.float32)
    img = img[:,:,::-1] # BGR to RGB
    img = np.power(img,(1.0/2.2))
    img = cv2.resize(img, (320,240))
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    from diffihistogram import *
    from diffihistogram_wrap import *
    from showhistogram import *
    img_t=torch.from_numpy(img.copy()).view(1,img.shape[0],img.shape[1],img.shape[2])
    uv=rgb_to_uv(img_t)
    #hist_uv=Histogram2D(bins=64,min=0,max=0.4, norm=True, sigma=3, cuda=True)
    hist_uv=Histogram2D_wrap(bins=64,min=0.25, max=0.35, norm=True, cuda=True)
    out= hist_uv.forward(uv.view(uv.shape[0], uv.shape[1]*uv.shape[2], uv.shape[3]))
    # x  =hist_uv.centers.cpu().numpy()
    # y  =x
    # z  =out[0,:,:].cpu().detach().numpy()
    # #show_histogram_1d(x,y,z)
    # show_histogram_2d(x,y,z)


    rgb=uv_to_rgb(uv).numpy()
    print('rgb', rgb.shape)
    import matplotlib.pyplot as plt
    plt.imshow(rgb[0,:,:,:])
    plt.show()
