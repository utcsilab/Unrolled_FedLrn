#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:15:18 2021

@author: brettlevac
"""
import h5py, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from resnet import ResNet
import glob
from deepinpy.opt import opt
from tqdm import tqdm
import torch.optim as optim
from dotmap import DotMap
from B_D_FastMRI import BrainFastMRI_Denoising

def loss_fun(pred, gt):
    resid = pred - gt
    #return torch.mean(torch.sum(torch.real(torch.conj(resid) * resid)))
    return torch.mean(torch.real(opt.zdot_single_batch(resid)))


#Define training parameters
train_param = DotMap()
train_param.global_steps = 10
train_param.global_update_mode = 'Full_avg' #'Full_avg' replaces entire local nets with update, 'Subset_avg' = subset of global net is shared between local nets
#add flexability for specifying how much of the network is averaged

#Define local Network Params
hparams = DotMap()
hparams.img_arch = 'ResNet'
hparams.img_channels = 18 if hparams.img_arch == 'UNet' else 64
hparams.img_blocks = 4


#Data Loader setup
ksp_maps_dir_1 = '/csiNAS/mridata/fastmri_brain/multicoil_val_espiritWc0_mvue_320/'
train_files_1  = sorted(glob.glob(ksp_maps_dir_1 + '/*.h5'))
train_dataset_1 = BrainFastMRI_Denoising(train_files_1, stdev = 0.0001)
train_loader_1  = DataLoader(train_dataset_1, batch_size=10, shuffle=True, num_workers=16, drop_last=True)

ksp_maps_dir_2 = '/csiNAS/mridata/fastmri_brain/multicoil_val_espiritWc0_mvue_320/'
train_files_2  = sorted(glob.glob(ksp_maps_dir_2 + '/*.h5'))
train_dataset_2 = BrainFastMRI_Denoising(train_files_2, stdev = 0.0005)
train_loader_2  = DataLoader(train_dataset_2, batch_size=10, shuffle=True, num_workers=16, drop_last=True)




if hparams.img_arch =='ResNet':
    image_net_1 = ResNet(in_channels=2, latent_channels=hparams.img_channels, num_blocks=hparams.img_blocks,kernel_size=3, batch_norm=False)
    image_net_2 = ResNet(in_channels=2, latent_channels=hparams.img_channels, num_blocks=hparams.img_blocks,kernel_size=3, batch_norm=False)
    global_net = ResNet(in_channels=2, latent_channels=hparams.img_channels, num_blocks=hparams.img_blocks,kernel_size=3, batch_norm=False)

    
#Set Optimizer Parameters
lr = 1
optimizer_1 = optim.SGD(image_net_1.parameters(), lr=lr)
optimizer_2 = optim.SGD(image_net_2.parameters(), lr=lr)

#set global update counts
num_epoch = 10


for epoch_idx in range(num_epoch):
    
    
    
    #update local net #1
    for idx, sample in tqdm(enumerate(train_loader_1)):
        
        #zero gradient for 1st net
        optimizer_1.zero_grad()
        
        #load noise and GT images for both Nets
        gt_img_1 = torch.reshape(sample['gt_img'],(sample['gt_img'].shape[0]*sample['gt_img'].shape[1],sample['gt_img'].shape[2],sample['gt_img'].shape[3] ))
        noise_img_1 = torch.reshape(sample['noise_img'],(sample['noise_img'].shape[0]*sample['noise_img'].shape[1],sample['noise_img'].shape[2],sample['noise_img'].shape[3] ))
        
        #calc Network output and loss for 1st local net
        denoise_img_1 = image_net_1(noise_img_1)    
        loss_1 = loss_fun(gt_img_1,denoise_img_1) 
        
        # Backprop
        loss_1.backward()
        # Verbose
        print('Epoch %d, Step %d, Batch loss Net1 %.4f.' % (epoch_idx, idx, loss_1.item()))
        
        # Optimizer Step
        optimizer_1.step()   
        
        
        #update local net #2
    for idx, sample in tqdm(enumerate(train_loader_2)):
        
        #zero gradient for 2nd net
        optimizer_2.zero_grad()
        
        gt_img_2 = torch.reshape(sample['gt_img'],(sample['gt_img'].shape[0]*sample['gt_img'].shape[1],sample['gt_img'].shape[2],sample['gt_img'].shape[3] ))
        noise_img_2 = torch.reshape(sample['noise_img'],(sample['noise_img'].shape[0]*sample['noise_img'].shape[1],sample['noise_img'].shape[2],sample['noise_img'].shape[3] ))
        
        #calc Network output and loss for 2nd local net        
        denoise_img_2 = image_net_2(noise_img_2)    
        loss_2 = loss_fun(gt_img_2,denoise_img_2)    
        
        # Backprop
        loss_2.backward()
        # Verbose
        print('Epoch %d, Step %d, Batch loss Net2 %.4f' % (epoch_idx, idx, loss_2.item()))
        
        # Optimizer Step  
        optimizer_2.step()   
        
        
        
        
        
    #average network weights
    sdA = image_net_1.state_dict()
    sdB = image_net_2.state_dict()

    # Average all parameters for global network
    for key in sdA:
        sdB[key] = (sdB[key] + sdA[key]) / 2.

    # set weights in global network
    global_net.load_state_dict(sdB)
    
    
    #set weights in local networks
    if train_param.global_update_mode =='Full_avg':
        image_net_1.load_state_dict(global_net)
        image_net_2.load_state_dict(global_net)
        
    #add capabilities for subsets of weights to be shared