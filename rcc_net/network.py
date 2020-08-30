import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import sys
sys.path.append('./auxiliary/')
from utils import *
from model import CreateNet_AlexNet
from alexnet import *


#2d lstm
from convolution_lstm import *

#for visualization
import matplotlib.pyplot as plt

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


class CreateNet(nn.Module):
    def __init__(self,model):
        super(CreateNet,self).__init__()
        self.squeezenet1_1 = nn.Sequential(*list(model.children())[0][:12])
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(512, 64, kernel_size=6, stride=1,padding=3),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Sigmoid(),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,x):
        x = self.squeezenet1_1(x)
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

class CreateNet_rccnet_2squeeze_fcn_3stage(nn.Module):
    def __init__(self):
        super(CreateNet_rccnet_2squeeze_fcn_3stage,self).__init__()

        self.submodel1 = CreateNet(squeezenet1_1(pretrained=True))
        self.submodel2 = CreateNet(squeezenet1_1(pretrained=True))
        model1=squeezenet1_1(pretrained=True)
        model2=squeezenet1_1(pretrained=True)
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.squeezenet1_1_B = nn.Sequential(*list(model2.children())[0][:12])
        self.lstm_A=ConvLSTMCell(512,128,5)
        self.lstm_B=ConvLSTMCell(512,128,5)
        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(128*2, 64, kernel_size=6, stride=1,padding=3),
                  nn.Sigmoid(),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  nn.Sigmoid()
        )

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)

        #a line
        if True:
            x=a
            output1 = self.submodel1(x)
            pred1 = torch.sum(torch.sum(output1,2),2)
            pred1 = torch.nn.functional.normalize(pred1,dim=1)
            correct_img1 =  correct_image_nolinear(x,pred1)
            output2 = self.submodel2(correct_img1)
            pred2 = torch.sum(torch.sum(output2,2),2)
            pred2 = torch.nn.functional.normalize(pred2,dim=1)
            a = correct_image_nolinear(x,torch.mul(pred1,pred2))

            a_pred1=pred1.view(nbatch,nstep,-1)
            a_pred2=torch.mul(pred1,pred2).view(nbatch,nstep,-1)
            a_pred1=a_pred1[:,-1,:]
            a_pred2=a_pred2[:,-1,:]

        if True:
            x=b
            output1 = self.submodel1(x)
            pred1 = torch.sum(torch.sum(output1,2),2)
            pred1 = torch.nn.functional.normalize(pred1,dim=1)
            correct_img1 =  correct_image_nolinear(x,pred1)
            output2 = self.submodel2(correct_img1)
            pred2 = torch.sum(torch.sum(output2,2),2)
            pred2 = torch.nn.functional.normalize(pred2,dim=1)
            b = correct_image_nolinear(x,torch.mul(pred1,pred2))

            b_pred1=pred1.view(nbatch,nstep,-1)
            b_pred2=torch.mul(pred1,pred2).view(nbatch,nstep,-1)
            b_pred1=torch.mean(b_pred1,1)
            b_pred2=torch.mean(b_pred2,1)


        a = self.squeezenet1_1_A(a)
        b = self.squeezenet1_1_B(b)
        _,nchan_a,h_a,w_a = a.shape
        _,nchan_b,h_b,w_b = b.shape
        a=a.view(nbatch, nstep, nchan_a, h_a, w_a)
        b=b.view(nbatch, nstep, nchan_b, h_b, w_b)

        self.lstm_A.init_hidden(nbatch, 128, [h_a, w_a])
        self.lstm_B.init_hidden(nbatch, 128, [h_b, w_b])
        hidden_state_A=torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        hidden_state_B=torch.zeros((nbatch, 128, h_b, w_b)).cuda()
        cell_state_B  =torch.zeros((nbatch, 128, h_b, w_b)).cuda()

        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], hidden_state_A, cell_state_A)
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], hidden_state_B, cell_state_B)
        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)

        # print('a_pred1, a_pred2', a_pred1, a_pred2)
        # print('b_pred1, b_pred2', b_pred1, b_pred2)

        return c,(a_pred1, a_pred2),(b_pred1,b_pred2)

class CreateNet_rccnet_2squeeze_fcn_hiddenN_kernelK(nn.Module):
    def __init__(self,model1, model2, lstm_hiddendim, lstm_ksize):
        super(CreateNet_rccnet_2squeeze_fcn_hiddenN_kernelK,self).__init__()
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.squeezenet1_1_B = nn.Sequential(*list(model2.children())[0][:12])

        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_hiddendim=lstm_hiddendim
        self.lstm_ksize=lstm_ksize

        self.lstm_A=ConvLSTMCell(512,self.lstm_hiddendim,self.lstm_ksize)
        self.lstm_B=ConvLSTMCell(512,self.lstm_hiddendim,self.lstm_ksize)


        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(self.lstm_hiddendim*2, self.lstm_hiddendim, kernel_size=6, stride=1,padding=3),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Sigmoid(),
                  nn.Conv2d(self.lstm_hiddendim, 3, kernel_size=1, stride=1),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)
        a = self.squeezenet1_1_A(a)
        b = self.squeezenet1_1_B(b)
        _,nchan_a,h_a,w_a = a.shape
        _,nchan_b,h_b,w_b = b.shape
        a=a.view(nbatch, nstep, nchan_a, h_a, w_a)
        b=b.view(nbatch, nstep, nchan_b, h_b, w_b)
        # a = torch.mean(a,dim=(2,3))
        # b = torch.mean(b,dim=(2,3))
        # a=a.view(nbatch,nstep,-1)
        # b=b.view(nbatch,nstep,-1)

        self.lstm_A.init_hidden(nbatch, self.lstm_hiddendim, [h_a, w_a])
        self.lstm_B.init_hidden(nbatch, self.lstm_hiddendim, [h_b, w_b])
        hidden_state_A=torch.zeros((nbatch, self.lstm_hiddendim, h_a, w_a)).cuda()
        cell_state_A  =torch.zeros((nbatch, self.lstm_hiddendim, h_a, w_a)).cuda()
        hidden_state_B=torch.zeros((nbatch, self.lstm_hiddendim, h_b, w_b)).cuda()
        cell_state_B  =torch.zeros((nbatch, self.lstm_hiddendim, h_b, w_b)).cuda()

        #a,(h_a,_)=self.lstm_A(a)
        #b,(h_b,_)=self.lstm_B(b)
        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], hidden_state_A, cell_state_A)
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], hidden_state_B, cell_state_B)

        #print('hidden_state_A', hidden_state_A.shape)
        #print('hidden_state_B', hidden_state_B.shape)
        #c=torch.cat((h_a,h_b),2) #(1, batch, channel)
        #c=c[-1,:,:]
        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)
        return c

class CreateNet_rccnet_2squeeze_fcn(nn.Module):
    def __init__(self,model1, model2):
        super(CreateNet_rccnet_2squeeze_fcn,self).__init__()
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.squeezenet1_1_B = nn.Sequential(*list(model2.children())[0][:12])

        # self.lstm_A=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        # self.lstm_B=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A=ConvLSTMCell(512,128,5)
        self.lstm_B=ConvLSTMCell(512,128,5)


        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(128*2, 64, kernel_size=6, stride=1,padding=3),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Sigmoid(),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)
        a = self.squeezenet1_1_A(a)
        b = self.squeezenet1_1_B(b)
        _,nchan_a,h_a,w_a = a.shape
        _,nchan_b,h_b,w_b = b.shape
        a=a.view(nbatch, nstep, nchan_a, h_a, w_a)
        b=b.view(nbatch, nstep, nchan_b, h_b, w_b)
        # a = torch.mean(a,dim=(2,3))
        # b = torch.mean(b,dim=(2,3))
        # a=a.view(nbatch,nstep,-1)
        # b=b.view(nbatch,nstep,-1)

        self.lstm_A.init_hidden(nbatch, 128, [h_a, w_a])
        self.lstm_B.init_hidden(nbatch, 128, [h_b, w_b])
        hidden_state_A=torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        hidden_state_B=torch.zeros((nbatch, 128, h_b, w_b)).cuda()
        cell_state_B  =torch.zeros((nbatch, 128, h_b, w_b)).cuda()

        #a,(h_a,_)=self.lstm_A(a)
        #b,(h_b,_)=self.lstm_B(b)
        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], hidden_state_A, cell_state_A)
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], hidden_state_B, cell_state_B)

        a_show, b_show = a[:,t,:], b[:,t,:]
        # print('a_show', a_show.shape)
        # img_show=a_show[0,0,:,:]
        # print('hidden_state_A', hidden_state_A.shape)
        # img_show=hidden_state_A[0,0,:,:]
        # img_show=img_show/img_show.max()
        # plt.imshow(img_show.detach().cpu().numpy());
        # plt.axis('off')
        # plt.show()


        #print('hidden_state_A', hidden_state_A.shape)
        #print('hidden_state_B', hidden_state_B.shape)
        #c=torch.cat((h_a,h_b),2) #(1, batch, channel)
        #c=c[-1,:,:]
        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)
        return c

class CreateNet_rccnet_2squeeze_fcn_savefeature(nn.Module):
    def __init__(self,model1, model2):
        super(CreateNet_rccnet_2squeeze_fcn_savefeature,self).__init__()
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.squeezenet1_1_B = nn.Sequential(*list(model2.children())[0][:12])

        # self.lstm_A=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        # self.lstm_B=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A=ConvLSTMCell(512,128,5)
        self.lstm_B=ConvLSTMCell(512,128,5)


        self.fc = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                  nn.Conv2d(128*2, 64, kernel_size=6, stride=1,padding=3),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Sigmoid(),
                  nn.Conv2d(64, 3, kernel_size=1, stride=1),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )
        self.save_feature=[]

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)
        a = self.squeezenet1_1_A(a)
        b = self.squeezenet1_1_B(b)
        _,nchan_a,h_a,w_a = a.shape
        _,nchan_b,h_b,w_b = b.shape
        a=a.view(nbatch, nstep, nchan_a, h_a, w_a)
        b=b.view(nbatch, nstep, nchan_b, h_b, w_b)
        # a = torch.mean(a,dim=(2,3))
        # b = torch.mean(b,dim=(2,3))
        # a=a.view(nbatch,nstep,-1)
        # b=b.view(nbatch,nstep,-1)

        self.lstm_A.init_hidden(nbatch, 128, [h_a, w_a])
        self.lstm_B.init_hidden(nbatch, 128, [h_b, w_b])
        hidden_state_A=torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128, h_a, w_a)).cuda()
        hidden_state_B=torch.zeros((nbatch, 128, h_b, w_b)).cuda()
        cell_state_B  =torch.zeros((nbatch, 128, h_b, w_b)).cuda()

        #a,(h_a,_)=self.lstm_A(a)
        #b,(h_b,_)=self.lstm_B(b)
        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], hidden_state_A, cell_state_A)
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], hidden_state_B, cell_state_B)

        a_show, b_show = a[:,t,:], b[:,t,:]

        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)

        self.save_feature=[]
        self.save_feature.append(a_show)
        self.save_feature.append(hidden_state_A)
        return c

    def get_feature(self):
        return self.save_feature

class CreateNet_rccnet_1squeeze(nn.Module):
    def __init__(self,model1):
        super(CreateNet_rccnet_1squeeze,self).__init__()
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])

        # self.lstm_A=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        # self.lstm_B=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A=nn.LSTMCell(512,128)

        self.fc = nn.Sequential(
                  #nn.Dropout(p=0.5),
                  nn.Linear(128,64),
                  nn.Sigmoid(),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Linear(64,3),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,a):
        assert len(a.shape)==5
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        a = self.squeezenet1_1_A(a)
        a = torch.mean(a,dim=(2,3))
        a=a.view(nbatch,nstep,-1)

        hidden_state_A=torch.zeros((nbatch, 128)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128)).cuda()

        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], (hidden_state_A, cell_state_A))

        #c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=hidden_state_A
        c=self.fc(c)
        return c

class CreateNet_rccnet_2squeeze(nn.Module):
    def __init__(self,model1,model2):
        super(CreateNet_rccnet_2squeeze,self).__init__()
        self.squeezenet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.squeezenet1_1_B = nn.Sequential(*list(model2.children())[0][:12])

        # self.lstm_A=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        # self.lstm_B=nn.LSTM(input_size=512,hidden_size=128,num_layers=1,
        #                     batch_first=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A=nn.LSTMCell(512,128)
        self.lstm_B=nn.LSTMCell(512,128)

        self.fc = nn.Sequential(
                  #nn.Dropout(p=0.5),
                  nn.Linear(256,64),
                  nn.Sigmoid(),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Linear(64,3),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)
        a = self.squeezenet1_1_A(a)
        b = self.squeezenet1_1_B(b)
        a = torch.mean(a,dim=(2,3))
        b = torch.mean(b,dim=(2,3))
        a=a.view(nbatch,nstep,-1)
        b=b.view(nbatch,nstep,-1)

        hidden_state_A=torch.zeros((nbatch, 128)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128)).cuda()
        hidden_state_B=torch.zeros((nbatch, 128)).cuda()
        cell_state_B  =torch.zeros((nbatch, 128)).cuda()

        #a,(h_a,_)=self.lstm_A(a)
        #b,(h_b,_)=self.lstm_B(b)
        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], (hidden_state_A, cell_state_A))
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], (hidden_state_B, cell_state_B))

        #c=torch.cat((h_a,h_b),2) #(1, batch, channel)
        #c=c[-1,:,:]
        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)
        return c

class CreateNet_rccnet_2alexnet(nn.Module):
    def __init__(self,model1,model2):
        super(CreateNet_rccnet_2alexnet,self).__init__()
        self.alexnet1_1_A = nn.Sequential(*list(model1.children())[0][:12])
        self.alexnet1_1_B = nn.Sequential(*list(model2.children())[0][:12])

        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.lstm_A=nn.LSTMCell(256,128)
        self.lstm_B=nn.LSTMCell(256,128)

        self.fc = nn.Sequential(
                  #nn.Dropout(p=0.5),
                  nn.Linear(256,64),
                  nn.Sigmoid(),
                  #nn.ReLU(inplace=True),
                  #nn.Dropout(p=0.5),
                  nn.Linear(64,3),
                  #nn.ReLU(inplace=True)
                  nn.Sigmoid()
        )

    def forward(self,a,b):
        assert len(a.shape)==5
        assert len(a.shape)==len(b.shape)
        ndims = len(a.shape)
        nbatch,nstep,nchan,h,w=a.shape
        a=a.view(nbatch*nstep,nchan,h,w)
        b=b.view(nbatch*nstep,nchan,h,w)
        a = self.alexnet1_1_A(a)
        b = self.alexnet1_1_B(b)
        a = torch.mean(a,dim=(2,3))
        b = torch.mean(b,dim=(2,3))
        a=a.view(nbatch,nstep,-1)
        b=b.view(nbatch,nstep,-1)

        hidden_state_A=torch.zeros((nbatch, 128)).cuda()
        cell_state_A  =torch.zeros((nbatch, 128)).cuda()
        hidden_state_B=torch.zeros((nbatch, 128)).cuda()
        cell_state_B  =torch.zeros((nbatch, 128)).cuda()

        #a,(h_a,_)=self.lstm_A(a)
        #b,(h_b,_)=self.lstm_B(b)
        for t in range(a.shape[1]):
            hidden_state_A, cell_state_A=self.lstm_A(a[:,t,:], (hidden_state_A, cell_state_A))
            hidden_state_B, cell_state_B=self.lstm_B(b[:,t,:], (hidden_state_B, cell_state_B))

        #c=torch.cat((h_a,h_b),2) #(1, batch, channel)
        #c=c[-1,:,:]

        c=torch.cat((hidden_state_A, hidden_state_B),1)
        c=self.fc(c)
        return c

if __name__=='__main__':
    # nstep=10
    # SqueezeNet = squeezenet1_1(pretrained=True)
    # network = CreateNet_rccnet_2squeeze(SqueezeNet).cuda()
    # input = torch.randn([16,nstep,3,32,32]).cuda()
    # label = torch.randn([16,3]).cuda()
    # pred = network(input, input)
    # print(pred,pred.shape)

    nstep=10
    SqueezeNet = squeezenet1_1(pretrained=True)
    network = CreateNet_rccnet_2squeeze_fcn(SqueezeNet).cuda()
    input = torch.randn([1,nstep,3,224,224]).cuda()
    label = torch.randn([1,3]).cuda()
    pred = network(input, input)
    print(pred,pred.shape)
