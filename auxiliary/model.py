import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from utils import *


from torch.distributions import Categorical

from squeezenet import *
from alexnet import *
from vggnet import *

class CreateNet_hist(nn.Module):
    def __init__(self,model):
        super(CreateNet_hist,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(512, 64, kernel_size=6, stride=1,padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(64, 2, kernel_size=1, stride=1),
                  nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.squeezenet1_1(x)
        x = self.fc(x) # (1,2,16,16)
        return x

class CreateNet(nn.Module):
    def __init__(self,model):
        super(CreateNet,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(512, 64, kernel_size=6, stride=1,padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.squeezenet1_1(x)
        x = self.fc(x)
        return x

class CreateNet_uv(nn.Module):
    def __init__(self,model):
        super(CreateNet_uv,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(512, 64, kernel_size=6, stride=1,padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )

    def forward(self,x):
        x = self.squeezenet1_1(x)
        x = self.fc(x)
        return x

class CreateNet_AlexNet(nn.Module):
    def __init__(self,model):
        super(CreateNet_AlexNet,self).__init__()
        self.alexnet = nn.Sequential(*list(model.children())[0][:12])
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(256, 64, kernel_size=6, stride=1,padding=3),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.alexnet(x)
        x = self.fc(x)
        return x


class CreateNet_3stage(nn.Module):
    def __init__(self,num_model=2):
        super(CreateNet_3stage,self).__init__()
        self.submodel1 = CreateNet(squeezenet1_1(pretrained=True))
        self.submodel2 = CreateNet(squeezenet1_1(pretrained=True))
        self.submodel3 = CreateNet(squeezenet1_1(pretrained=True))

    def forward(self,x):#x[bs,3,h,w]
        output1 = self.submodel1(x)
        pred1 = torch.sum(torch.sum(output1,2),2)

        pred1 = torch.nn.functional.normalize(pred1,dim=1)
        correct_img1 =  correct_image_nolinear(x,pred1)
        output2 = self.submodel2(correct_img1)
        pred2 = torch.sum(torch.sum(output2,2),2)
        pred2 = torch.nn.functional.normalize(pred2,dim=1)
        correct_img2 = correct_image_nolinear(x,torch.mul(pred1,pred2))
        output3 = self.submodel3(correct_img2)
        pred3 = torch.sum(torch.sum(output3,2),2)
        pred3 = torch.nn.functional.normalize(pred3,dim=1)
        return pred1,pred2,pred3



if __name__=='__main__':
    #test squeezenet
    # SqueezeNet = squeezenet1_1(pretrained=True)
    # network = CreateNet(SqueezeNet).cuda()

    #test alexnet
    #AlexNet=alexnet(pretrained=True)
    #network=CreateNet_AlexNet(AlexNet).cuda()

    #test vggnet
    VggNet=vgg19(pretrained=True).cuda()
    print(VggNet.children())
    network=nn.Sequential(*list(VggNet.children())[0][:28])

    input = torch.randn([16,3,64,64]).cuda()
    label = torch.randn([16,3]).cuda()
    act = network(input)


    print(act.shape)
