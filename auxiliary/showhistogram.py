import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffihistogram import *
TIME_PAUSE=1000

def show_histogram_1d(x,y,z):
    #x: x edges, 1d list
    #y: y edges, 1d list
    #z: point counts, 2d matrix
    X,Y = np.meshgrid(x,y)
    plt.imshow(z)
    plt.grid(False)
    plt.colorbar()
    plt.show()
    #plt.draw()
    # plt.pause(TIME_PAUSE)
    # plt.close()

def show_histogram_2d(x,y,z):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')
    X,Y= np.meshgrid(x, y)
    Z = z
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim(0, np.max(Z))
    #ax.set_zlim(0, 0.005)
    ax.zaxis.set_major_locator(plt.LinearLocator(10))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.06f'))
    fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
    plt.show()
    #plt.draw()
    # plt.pause(TIME_PAUSE)
    # plt.close()

if __name__=='__main__':
    print('demo showhistogram')
    nbatch=2
    data = 64*torch.randn(nbatch*240*320*2).view(nbatch,-1, 2).cuda()
    data.requires_grad=True
    hist2d=Histogram2D(bins=32,min=0,max=64, norm=True, sigma=3, cuda=True)
    out   =hist2d.forward(data)
    x     =hist2d.centers.cpu().numpy()
    y     =x
    z     =out[0,:,:].cpu().detach().numpy()
    show_histogram_1d(x,y,z)
    #show_histogram_2d(x,y,z)
