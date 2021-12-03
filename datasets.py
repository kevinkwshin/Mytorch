import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import os
import glob
from natsort import natsorted

import numpy as np
import torch

import mclahe
import imageio

import cv2
import mclahe
import imageio

from PIL import Image
from skimage.filters import frangi
import skimage.morphology
from skimage.morphology import white_tophat,black_tophat
from skimage.morphology import disk, star, square
from skimage.morphology import skeletonize, medial_axis, dilation, disk, remove_small_objects
from skimage.filters import *

import kornia
from kornia.morphology import *
from kornia.enhance import *

import matplotlib.pyplot as plt
import albumentations as albu


def clahe(img,adaptive_hist_range=False):
    """
    input 1 numpy shape image ( H x W x C)
    """
    temp = np.zeros_like(img)
    for idx in range(temp.shape[-1]):
        temp[...,idx] = mclahe.mclahe(img[...,idx], adaptive_hist_range=adaptive_hist_range)
    return temp

# Data Module
class dataset_CLAHE():
    def __init__(self,data_root='path/to/data', dataset_type='train', transform_spatial=None, transform=None, adaptive_hist_range=False):
        
        self.data_root = data_root
        if dataset_type =='train':
            self.x_list = natsorted(glob.glob(data_root+'/x_train/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_train/*'))
        elif dataset_type =='valid':
            self.x_list = natsorted(glob.glob(data_root+'/x_valid/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_valid/*'))
        elif dataset_type =='test':
            self.x_list = natsorted(glob.glob(data_root+'/x_test/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_test/*'))
        elif dataset_type =='etest':
            self.x_list = natsorted(glob.glob(data_root+'/x_etest/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_etest/*'))
        elif dataset_type =='infernce':
            self.x_list = natsorted(glob.glob(data_root+'/x_infernce/*'))
            self.y_list = self.x_list
                
        self.transform = transform
        self.transform_spatial = transform_spatial
        self.adaptive_hist_range = adaptive_hist_range
        print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, idx):
        # load data
        fname = self.x_list[idx]
        if fname.split('.')[-1] == 'npy':
            x = np.load(self.x_list[idx])
            y = np.load(self.y_list[idx])
            x = np.moveaxis(x,0,-1)
            y = np.moveaxis(y,0,-1)
            x = x[:,:,0]
            x = np.expand_dims(x,-1)
            y = y[:,:,0]
            y = np.expand_dims(y,-1)
            if np.max(x)<=1:
                x = x*255
            x = x.astype(np.uint8)
        else:
            x = cv2.imread(self.x_list[idx])
            y = cv2.imread(self.y_list[idx])
            if self.x_list[idx] == self.y_list[idx]:
                mask = 0 # infernce
            else:
                mask = 0 if 23 in y else 1 # psuedo
            x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
            
        # shape check, y.shape == (h,w,1)
        if len(y.shape)==2:
            y = np.expand_dims(y,-1)
        elif len(y.shape)==3 and y.shape[-1]==3:
            y = np.expand_dims(y[...,0],-1)

        # color filter warp augmentation
        if self.transform:
            sample = self.transform(image = x, mask = y)
            x, y= sample['image'], sample['mask']

        # normalization
        x = x.astype(np.float32)
        x = clahe(x,self.adaptive_hist_range) # 0 ~ 1
        
        # spatial transform
        if self.transform_spatial:
            sample = self.transform_spatial(image = x, mask = y)
            x, y = sample['image'], sample['mask']
        
        # # inverse weight of y
        # skeleton,distance = medial_axis(y[...,0], return_distance=True)
        # distance[distance==0]=1000
        # y = np.expand_dims(np.exp(-distance/16),-1)
        
        # to tensor
        x = np.moveaxis(x,-1,0).astype(np.float32)
        y = np.moveaxis(y,-1,0).astype(np.float32)

        x = torch.tensor(x) # 0 ~ 1
        y = torch.tensor(y) # 0 ~ 4095
        y = y[0].unsqueeze(0)
        return {'x':x, 'y':y, 'fname':fname, 'mask':mask}

# Data Module
class dataset_ACLAHE():
    def __init__(self,data_root='path/to/data', dataset_type='train', transform_spatial=None, transform=None, adaptive_hist_range=True):
        
        self.data_root = data_root
        if dataset_type =='train':
            self.x_list = natsorted(glob.glob(data_root+'/x_train/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_train/*'))
        elif dataset_type =='valid':
            self.x_list = natsorted(glob.glob(data_root+'/x_valid/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_valid/*'))
        elif dataset_type =='test':
            self.x_list = natsorted(glob.glob(data_root+'/x_test/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_test/*'))
        elif dataset_type =='etest':
            self.x_list = natsorted(glob.glob(data_root+'/x_etest/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_etest/*'))
        elif dataset_type =='infernce':
            self.x_list = natsorted(glob.glob(data_root+'/x_infernce/*'))
            self.y_list = self.x_list
                
        self.transform = transform
        self.transform_spatial = transform_spatial
        self.adaptive_hist_range = adaptive_hist_range
        print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, idx):
        # load data
        fname = self.x_list[idx]
        if fname.split('.')[-1] == 'npy':
            x = np.load(self.x_list[idx])
            y = np.load(self.y_list[idx])
            x = np.moveaxis(x,0,-1)
            y = np.moveaxis(y,0,-1)
            x = x[:,:,0]
            x = np.expand_dims(x,-1)
            y = y[:,:,0]
            y = np.expand_dims(y,-1)
            if np.max(x)<=1:
                x = x*255
            x = x.astype(np.uint8)
        else:
            x = cv2.imread(self.x_list[idx])
            y = cv2.imread(self.y_list[idx])
            if self.x_list[idx]==self.y_list[idx]:
                mask = 0 # infernce
            else:
                mask = 0 if 23 in y else 1 # psuedo
            x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
            
        # shape check, y.shape == (h,w,1)
        if len(y.shape)==2:
            y = np.expand_dims(y,-1)
        elif len(y.shape)==3 and y.shape[-1]==3:
            y = np.expand_dims(y[...,0],-1)

        # color filter warp augmentation
        if self.transform:
            sample = self.transform(image = x, mask = y)
            x, y= sample['image'], sample['mask']

        # normalization
        x = x.astype(np.float32)
        x = cv2.addWeighted(x, 4, cv2.GaussianBlur(x, (0,0), x.shape[0]/2/30), -4, x.shape[0]/4)
        x = clahe(x, self.adaptive_hist_range) # 0 ~ 1
        
        # spatial transform
        if self.transform_spatial:
            sample = self.transform_spatial(image = x, mask = y)
            x, y = sample['image'], sample['mask']
        
        # to tensor
        x = np.moveaxis(x,-1,0).astype(np.float32)
        y = np.moveaxis(y,-1,0).astype(np.float32)

        x = torch.tensor(x) # 0 ~ 1
        y = torch.tensor(y) # 0 ~ 4095
        y = y[0].unsqueeze(0)
        return {'x':x, 'y':y, 'fname':fname, 'mask':mask}
    
# Data Module
class dataset_CE():
    def __init__(self,data_root='path/to/data', dataset_type='train', transform_spatial=None, transform=None, adaptive_hist_range=False):
        
        self.data_root = data_root
        if dataset_type =='train':
            self.x_list = natsorted(glob.glob(data_root+'/x_train/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_train/*'))
        elif dataset_type =='valid':
            self.x_list = natsorted(glob.glob(data_root+'/x_valid/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_valid/*'))
        elif dataset_type =='test':
            self.x_list = natsorted(glob.glob(data_root+'/x_test/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_test/*'))
        elif dataset_type =='etest':
            self.x_list = natsorted(glob.glob(data_root+'/x_etest/*'))
            self.y_list = natsorted(glob.glob(data_root+'/y_etest/*'))
        elif dataset_type =='infernce':
            self.x_list = natsorted(glob.glob(data_root+'/x_infernce/*'))
            self.y_list = self.x_list
                
        self.transform = transform
        self.transform_spatial = transform_spatial
        self.adaptive_hist_range = adaptive_hist_range
        print('total counts of dataset x {}, y {}'.format(len(self.x_list),len(self.y_list)))
        
    def __len__(self):
        return len(self.x_list)
  
    def __getitem__(self, idx):
        # load data
        fname = self.x_list[idx]
        if fname.split('.')[-1] == 'npy':
            x = np.load(self.x_list[idx])
            y = np.load(self.y_list[idx])
            x = np.moveaxis(x,0,-1)
            y = np.moveaxis(y,0,-1)
            x = x[:,:,0]
            x = np.expand_dims(x,-1)
            y = y[:,:,0]
            y = np.expand_dims(y,-1)
            if np.max(x)<=1:
                x = x*255
            x = x.astype(np.uint8)
        else:
            x = cv2.imread(self.x_list[idx])
            y = cv2.imread(self.y_list[idx])
            if self.x_list[idx]==self.y_list[idx]:
                mask = 0 # infernce
            else:
                mask = 0 if 23 in y else 1 # psuedo
            x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
            
        # shape check, y.shape == (h,w,1)
        if len(y.shape)==2:
            y = np.expand_dims(y,-1)
        elif len(y.shape)==3 and y.shape[-1]==3:
            y = np.expand_dims(y[...,0],-1)

        # color filter warp augmentation
        if self.transform:
            sample = self.transform(image = x, mask = y)
            x, y= sample['image'], sample['mask']

        # normalization
        x = x.astype(np.float32)
        x = cv2.addWeighted(x, 4, cv2.GaussianBlur(x, (0,0), x.shape[0]/2/30), -4, x.shape[0]/4)
        x = clahe(x, self.adaptive_hist_range) # 0 ~ 1
        
        # spatial transform
        if self.transform_spatial:
            sample = self.transform_spatial(image = x, mask = y)
            x, y = sample['image'], sample['mask']
        
        # to tensor
        x = np.moveaxis(x,-1,0).astype(np.float32)
        y = np.moveaxis(y,-1,0).astype(np.float32)

        x = torch.tensor(x) # 0 ~ 1
        y = torch.tensor(y) # 0 ~ 4095
        y = y[0].unsqueeze(0)
        return {'x':x, 'y':y, 'fname':fname, 'mask':mask}
    
# augmentation
def augmentation_imagesize(data_padsize=None, data_cropsize=None, data_resize=None, data_patchsize = None):
    """
    sizes should be in 
    """
    transform = list()
    
    if data_padsize:
        if len(data_padsize.split('_'))==1:
            data_padsize = int(data_padsize)
            transform.append(albu.PadIfNeeded(data_padsize, data_padsize, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True))
        else:
            data_padsize_h = int(data_padsize.split('_')[0])
            data_padsize_w = int(data_padsize.split('_')[1])
            transform.append(albu.PadIfNeeded(data_padsize_h, data_padsize_w, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True))
    if data_cropsize:
        if len(data_cropsize.split('_'))==1:
            data_cropsize = int(data_cropsize)
            transform.append(albu.CenterCrop(data_cropsize, data_cropsize, always_apply=True))
        else:
            data_cropsize_h = int(data_cropsize.split('_')[0])
            data_cropsize_w = int(data_cropsize.split('_')[1])
            transform.append(albu.CenterCrop(data_cropsize_h, data_cropsize_w, always_apply=True))
    if data_resize:
        if len(data_resize.split('_'))==1:
            data_resize = int(data_resize)
            transform.append(albu.Resize(data_resize, data_resize, interpolation=cv2.INTER_CUBIC, always_apply=True))
        else:
            data_resize_h = int(data_resize.split('_')[0])
            data_resize_w = int(data_resize.split('_')[1])
            transform.append(albu.Resize(data_resize_h, data_resize_w, interpolation=cv2.INTER_CUBIC, always_apply=True))
    if data_patchsize:
        if len(data_patchsize.split('_'))==1:
            data_patchsize = int(data_patchsize)
            transform.append(albu.RandomCrop(data_patchsize, data_patchsize, always_apply=True))
        else:
            data_patchsize_h = int(data_patchsize.split('_')[0])
            data_patchsize_w = int(data_patchsize.split('_')[1])
            transform.append(albu.RandomCrop(height=data_patchsize_h, width=data_patchsize_w, always_apply=True))
            
    return albu.Compose(transform,additional_targets={'mask_bg':'mask'})

def augmentation_train():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        
        albu.OneOf([
        albu.InvertImg(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=(-0.5, 0.3), contrast_limit=(-0.5, 0.3), brightness_by_max=False, p=0.5),
        albu.RandomGamma(gamma_limit=(50,120), p=.5),
        albu.RandomToneCurve(scale=0.4,p=.5),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=.5),
        albu.ChannelShuffle(p=.5),
        albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=.5),
        ],p=0.5),
                
        albu.OneOf([
        albu.RandomFog(fog_coef_lower=0.1, fog_coef_upper=.4, alpha_coef=0.06, p=0.5),
        albu.MotionBlur(blur_limit=7, p=0.5),
        albu.MedianBlur(blur_limit=7, p=0.5),
        albu.GlassBlur(sigma=0.5, max_delta=2, p=0.5),
        albu.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.1), p=0.5)
        ],p=0.5),
        
        albu.OneOf([
        albu.GaussNoise(var_limit=0.03, mean=0, p=0.5),
        albu.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.5),
        albu.ISONoise(color_shift=(0.01, 0.02),intensity=(0.1, 0.3),p=0.5),
        ],p=0.3),
        
        albu.OneOf([
        albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, alpha=1, sigma=50, alpha_affine=50, p=0.5),
        albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, distort_limit=(-0.3,0.3), num_steps=5, p=0.5),
        albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC, distort_limit=(-.05,.05), shift_limit=(-0.05,0.05), p=0.5),
        albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, shift_limit=(0.05,0.02), scale_limit=(-.1, 0), rotate_limit=2, p=0.5),   
        ],p=0.5),
                     
    ]
    return albu.Compose(train_transform)

def augmentation_valid():
    test_transform = [
        
    ]
    return albu.Compose(test_transform)