from __future__ import print_function
import os
import sys
import cv2
import random
import math
import time
import scipy.io
import numpy as np
import torch
import torch.utils.data as data

from config import *
from utils import rotate_and_crop

#for visualization
import matplotlib.pyplot as plt

import glob

class Temporal(data.Dataset):
    def __init__(self,train=True, onlylastimg=True,
                input_size=[FCN_INPUT_SIZE, FCN_INPUT_SIZE],
                mimic=False):
        self.benchmark='/home/qian2/qian2_old/CC_temporal_benchmark/Temporal_CC2/'
        self.train=train
        self.onlylastimg=onlylastimg
        self.input_size=input_size
        self.mimic=mimic

        if self.onlylastimg:
            if train:
                self.data_list=glob.glob(self.benchmark+'data/ndata_single/train*.npy')
                self.data_list.sort(key=lambda x:x.split('train')[-1][:-4])
            else:
                self.data_list=glob.glob(self.benchmark+'data/ndata_single/test*.npy')
                self.data_list.sort(key=lambda x:int(x.split('test')[-1][:-4]))
        else:
            if train:
                self.data_list=glob.glob(self.benchmark+'data/ndata/train*.npy')
                self.data_list.sort(key=lambda x:x.split('train')[-1][:-4])
            else:
                self.data_list=glob.glob(self.benchmark+'data/ndata/test*.npy')

                self.data_list.sort(key=lambda x:int(x.split('test')[-1][:-4]) )

    def __getitem__(self, index):
        imgs_path=self.data_list[index]
        if self.onlylastimg:
            if self.train:
                label_path=imgs_path.replace('ndata_single','nlabel')
            else:
                label_path=imgs_path.replace('ndata_single','nlabel')
        else:
            if self.train:
                label_path=imgs_path.replace('ndata','nlabel')
            else:
                label_path=imgs_path.replace('ndata','nlabel')

        imgs,label = np.load(imgs_path), np.load(label_path)
        #img=imgs[-1]
        img=imgs
        img, illums=np.array(img,dtype='float32'), np.array(label,dtype='float32')

        if self.onlylastimg:
            if self.mimic:
                img_mimic= self.augment_mimic(img)
                img_mimic=img_mimic.transpose(0,3,1,2)
                img_mimic=torch.from_numpy(img_mimic.copy())
            if self.train:
                img, illums = self.augment_train(img,illums)
                img = np.clip(img, 0.0, 255.0)
                img = img * (1.0 / 255)
                img = img[:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(2,0,1) # hwc to chw
                img = torch.from_numpy(img.copy())
                illums = torch.from_numpy(illums.copy())
            else:
                img = self.crop_test(img,illums)
                img = np.clip(img, 0.0, 255.0)
                img = img * (1.0 / 255)
                img = img[:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(2,0,1) # hwc to chw
                img = torch.from_numpy(img.copy())
                illums = torch.from_numpy(illums.copy())
                #illums = torch.nn.functional.normalize(illums)
                img = img.type(torch.FloatTensor)
        else:
            if self.mimic:
                img_mimic= self.augment_mimic(img)
                img_mimic=img_mimic.transpose(0,3,1,2)
                img_mimic=torch.from_numpy(img_mimic.copy())

            if self.train:
                img, illums, color_aug= self.augment_train_temporal(img,illums)
                img = np.clip(img, 0.0, 255.0)
                img = img * (1.0 / 255)
                img = img[:,:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(0,3,1,2) # hwc to chw
                img = torch.from_numpy(img.copy())
                illums = torch.from_numpy(illums.copy())
                if self.mimic:
                    color_aug_tensor=torch.from_numpy(np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32))
                    color_aug_tensor=color_aug_tensor.view(1,3,1,1)
                    img_mimic=torch.mul(img_mimic, color_aug_tensor)
            else:
                #img, illums = self.augment_train_temporal(img,illums)
                img = self.crop_test_temporal(img, illums)
                img = np.clip(img, 0.0, 255.0)
                img = img * (1.0 / 255)
                img = img[:,:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(0,3,1,2) # hwc to chw
                img = torch.from_numpy(img.copy())
                illums = torch.from_numpy(illums.copy())

        #print(img.max(), img_mimic.max())

        if self.mimic:
            return img, img_mimic, illums, imgs_path
        else:
            return img,illums,imgs_path

    def augment_mimic(self, ldr):
        if len(ldr.shape)==4:
            img=ldr[-1]
            nsteps=ldr.shape[0]
        if len(ldr.shape)==3:
            img=ldr
            nsteps=1

        def crop(img):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            img = img.astype(np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 255.0)
            #plt.imshow(new_image);plt.show()
            return new_image

        img_list, img_temp=[], img[:,:,::-1]*(1.0/255)
        for i in range(nsteps):
            angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
            scale = 0.95#math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
            s = int(round(min(img_temp.shape[:2]) * scale))
            s = min(max(s, 10), min(img_temp.shape[:2]))
            start_x = random.randrange(0, img_temp.shape[0] - s + 1)
            start_y = random.randrange(0, img_temp.shape[1] - s + 1)
            img_new=crop(img_temp)


            img_list.append(img_new)
            img_temp=img_new
        img_mimic=np.stack(img_list)
        return img_mimic

    def augment_train_temporal(self, ldr, illum):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
        s = int(round(min(ldr.shape[1:3]) * scale))
        s = min(max(s, 10), min(ldr.shape[1:3]))

        start_x = random.randrange(0, ldr.shape[1] - s + 1)
        start_y = random.randrange(0, ldr.shape[2] - s + 1)
        flip_lr = random.randint(0, 1) # Left-right flip?
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR

        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 255.0)
            new_illum = np.clip(new_illum, 0.01, 100)
            return new_image, new_illum[::-1]

        img_list, illu_list=[],[]
        for i in range(ldr.shape[0]):
            newimg, newillu=crop(ldr[i],illum)
            img_list.append(newimg)
            illu_list.append(newillu)

        imgs=np.stack(img_list)
        return imgs, illum, color_aug

    def augment_train(self,ldr, illum):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)
        flip_lr = random.randint(0, 1) # Left-right flip?
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR

        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 255.0)
            new_illum = np.clip(new_illum, 0.01, 100)
            return new_image, new_illum[::-1]
        return crop(ldr, illum)

    def crop_test(self,img,illums,scale=0.5):
        #img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = cv2.resize(img, (self.input_size[0], self.input_size[1]))
        return img
    def crop_test_temporal(self,img,illums,scale=0.5):
        #img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img_list=[]
        for i in range(img.shape[0]):
            img_list.append(cv2.resize(img[i], (self.input_size[0], self.input_size[1])))
        imgs=np.stack(img_list)
        return imgs

    def __len__(self):
        return(len(self.data_list))

class ColorChecker(data.Dataset):
    def __init__(self,train=True,folds_num=1,
                methodtag='ffcc',
                input_size=[FCN_INPUT_SIZE, FCN_INPUT_SIZE]
                ):
        list_path = './data/color_cheker_data_meta.txt'
        with open(list_path,'r') as f:
            self.all_data_list = f.readlines()
        self.data_list = []
        folds = scipy.io.loadmat('./data/folds.mat')
        if train:
            img_idx = folds['tr_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-1])
        else:
            img_idx = folds['te_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-1])
        self.train = train
        self.input_size=input_size
        self.methodtag=methodtag


    def __getitem__(self,index):
        model = self.data_list[index]
        illums = []
        # filename
        fn = model.strip().split(' ')[1]
        img = np.load('./data/ndata/'+fn+'.npy')
        illums = np.load('./data/nlabel/'+fn+'.npy')
        img = np.array(img,dtype='float32')
        illums = np.array(illums,dtype='float32')
        if self.train:
            if self.methodtag=='ffcc':
                #img, illums = self.augment_train(img,illums)
                img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
                img = np.clip(img, 0.0, 65535.0)
                img = img * (1.0 / 65535)
                img = img[:,:,::-1] # BGR to RGB
                kernel_absodev=-np.ones((3,3))
                kernel_absodev[1,1]=8
                kernel_absodev/=8
                img_absodev=np.abs(cv2.filter2D(img, -1, kernel_absodev))
                #img_absodev+=0.01
                img=np.stack((img, img))
            if self.methodtag=='c4':
                img, illums = self.augment_train(img,illums)
                img = np.clip(img, 0.0, 65535.0)
                img = img * (1.0 / 65535)
                img = img[:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(2,0,1) # hwc to chw
            img = torch.from_numpy(img.copy())
            illums = torch.from_numpy(illums.copy())

        else:
            if self.methodtag=='ffcc':
                img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
                img = np.clip(img, 0.0, 65535.0)
                img = img * (1.0 / 65535)
                img = img[:,:,::-1] # BGR to RGB
                kernel_absodev=-np.ones((3,3))
                kernel_absodev[1,1]=8
                kernel_absodev/=8
                img_absodev=np.abs(cv2.filter2D(img, -1, kernel_absodev))
                #img_absodev+=0.01
                img=np.stack((img, img))
            if self.methodtag=='c4':
                img = self.crop_test(img,illums)
                img = np.clip(img, 0.0, 65535.0)
                img = img * (1.0 / 65535)
                img = img[:,:,::-1] # BGR to RGB
                img = np.power(img,(1.0/2.2))
                img = img.transpose(2,0,1) # hwc to chw
            img = torch.from_numpy(img.copy())
            illums = torch.from_numpy(illums.copy())
            img = img.type(torch.FloatTensor)
        return img,illums,fn

    def augment_train(self,ldr, illum):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)
        flip_lr = random.randint(0, 1) # Left-right flip?
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR

        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 65535)
            new_illum = np.clip(new_illum, 0.01, 100)
            return new_image, new_illum[::-1]
        return crop(ldr, illum)

    def crop_test(self,img,illums,scale=0.5):
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    def __len__(self):
        return(len(self.data_list))

if __name__=='__main__':
    test_ColorChecker=False
    test_Temporal=True

    if test_Temporal:
        dataset = Temporal(train=True,onlylastimg=False,input_size=[224,224], mimic=True)
        dataload = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=int(1))
        for ep in range(10):
            time1 = time.time()
            for i, data in enumerate(dataload):
                img,img_mimic, ill,fn = data
                print(img.shape, img_mimic.shape,  ill, fn)
                exit()
            #     break
            # break
    #test ColorChecker
    if test_ColorChecker:
        dataset = ColorChecker(train=True)
        dataload = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=int(30))
        for ep in range(10):
            time1 = time.time()
            for i, data in enumerate(dataload):
                img,ill,fn = data
                print(img.shape,  ill, fn)
                break
            break
