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
from resnet import ResNet, ResNetSplit, ResNet1
import glob
import opt
from tqdm import tqdm
import torch.optim as optim
from dotmap import DotMap
from datagen_copy import MCFullFastMRI, MCFullFastMRIBrain
import torch.fft as torch_fft

def loss_fun(pred, gt):
    resid = pred - gt
    #return torch.mean(torch.sum(torch.real(torch.conj(resid) * resid)))
    return torch.mean(torch.real(opt.zdot_single_batch(resid)))

def adjoint_op(ksp,mask,maps):
        #args:
        #ksp = (slices, coils, H, W)
        #mask = (None, None, None, W)
        #maps = (slices, coils, H, W)
        
        #returns:
        #x_adj = (slices, H, W)

        # Multiply input with mask and pad
        mask_adj_ext = mask
        ksp_padded  = ksp * mask_adj_ext
        
        # Get image representation of ksp
        img_ksp = torch_fft.fftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm='ortho')
        img_ksp = torch_fft.ifftshift(img_ksp, dim=(-2, -1))
        
        # Pointwise complex multiply with complex conjugate maps
        mult_result = img_ksp * torch.conj(maps)
        
        # Sum on coil axis
        x_adj = torch.sum(mult_result, dim=1)
        
        return x_adj


# Mask configs
train_mask_params, val_mask_params = DotMap(), DotMap()
# 'Accel_only': accelerates in PE direction.
# 'Gauss': Gaussian undersampling on top of PE acceleration
train_mask_params.mask_mode       = 'Accel_only' 
train_mask_params.p_r             = 0.4 # Rho in SSDU
train_mask_params.num_theta_masks = 6 # Split theta into a subset and apply them round-robin
train_mask_params.theta_fraction  = 0.5 # Fraction of each mask in set, auto-adjusted to use everything

hparams = DotMap()    
hparams.downsample   = 4 # Training R
# Model
hparams.img_init           = 'estimated'
hparams.mps_kernel_shape   = [15, 15, 9] # Always 15 coils
hparams.batch_size   = 5 # !!! can only be 1 or num_slices !!!
# Flag for generating masks
mask_flag = True
#Define training parameters
global_steps = 10 #number of updates to globally shared weights
#Define local Network Params
hparams.img_arch = 'ResNet'
hparams.img_channels = 18 if hparams.img_arch == 'UNet' else 64
hparams.share_mode = 'Seperate' #'Full_avg' replaces entire local nets with update, 'Subset_avg' = subset of global net is shared between local nets
#add flexability for specifying how much of the network is averaged

#specify local source params
num_local = 2 #set num of sources

#Data Loader setup(provide dataloaders for each source)
ksp_dir_1 = '/csiNAS/mridata/fastmri_knee/multicoil_train/'
train_ksp_files_1  = sorted(glob.glob(ksp_dir_1 + '/*.h5'))
maps_dir_1 = '/csiNAS/mridata/fastmri_knee/multicoil_train_Wc0_Espirit_maps/'
train_maps_files_1  = sorted(glob.glob(maps_dir_1 + '/*.h5'))

train_ksp_files_1 = [train_ksp_files_1[idx] for idx in range(150)]
train_maps_files_1 = [train_maps_files_1[idx] for idx in range(150)]

num_slices_1 = 5
center_slice_1 = 15
train_dataset_1 = MCFullFastMRI(train_ksp_files_1, num_slices_1, center_slice_1,
                                  downsample=hparams.downsample,
                                  mps_kernel_shape=hparams.mps_kernel_shape,
                                  maps=train_maps_files_1, mask_params=train_mask_params)
train_loader_1  = DataLoader(train_dataset_1, batch_size=hparams.batch_size, 
                               shuffle=False, num_workers=16, drop_last=True)




ksp_dir_2 = '/csiNAS/mridata/fastmri_brain/brain_multicoil_train/multicoil_train'
train_ksp_files_2  = sorted(glob.glob(ksp_dir_2 + '/*.h5'))
train_ksp_files_2.remove('/csiNAS/mridata/fastmri_brain/brain_multicoil_train/multicoil_train/file_brain_AXFLAIR_200_6002503.h5')#remove bad file from list
maps_dir_2 = '/csiNAS/brett/fastmri_brain/espirit_maps'
num_slices_2 = 5
center_slice_2 = 3
train_maps_files_2  = sorted(glob.glob(maps_dir_2 + '/*.h5'))
train_ksp_files_2 = [train_ksp_files_2[idx] for idx in range(150)]
train_maps_files_2 = [train_maps_files_2[idx] for idx in range(150)]
train_dataset_2 = MCFullFastMRIBrain(train_ksp_files_2, num_slices_2, center_slice_2,
                                  downsample=hparams.downsample,
                                  mps_kernel_shape=hparams.mps_kernel_shape,
                                  maps=train_maps_files_2, mask_params=train_mask_params)
train_loader_2  = DataLoader(train_dataset_2, batch_size=hparams.batch_size, 
                               shuffle=False, num_workers=16, drop_last=True)

#Define network architectures
num_blocks1 = 2
num_blocks2 = 2

if hparams.img_arch =='ResNet':
    #need to modify Resnet definitions to return intermediate outputs for skip connections
    image_net_1 = ResNetSplit(num_blocks1=num_blocks1, num_blocks2=num_blocks2, in_channels=2, latent_channels=64, kernel_size=7, bias=False, batch_norm=False, dropout=0, topk=None, l1lam=None)
    image_net_2 = ResNetSplit(num_blocks1=num_blocks1, num_blocks2=num_blocks2, in_channels=2, latent_channels=64, kernel_size=7, bias=False, batch_norm=False, dropout=0, topk=None, l1lam=None)
    global_net = ResNetSplit(num_blocks1=num_blocks1, num_blocks2=num_blocks2, in_channels=2, latent_channels=64, kernel_size=7, bias=False, batch_norm=False, dropout=0, topk=None, l1lam=None)
    
    #image_net_1 = ResNet(num_blocks=6, in_channels=2, latent_channels=64, kernel_size=7, bias=False, batch_norm=False, dropout=0, topk=None, l1lam=None)
    #image_net_2 = ResNet(num_blocks=6, in_channels=2, latent_channels=64, kernel_size=7, bias=False, batch_norm=False, dropout=0, topk=None, l1lam=None)


#Set Optimizer Parameters
lr = 0.0001
optimizer_1 = optim.Adam(image_net_1.parameters(), lr=lr)
optimizer_2 = optim.Adam(image_net_2.parameters(), lr=lr)



#loss logger
train_loss_1 = []
train_loss_2 = []


# Global directory
global_dir = 'models/%s/%s/blocksplit%d%d' % (
    hparams.share_mode, hparams.img_arch, num_blocks1, num_blocks2)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)


#list to store local site elements
DL_list = []
opt_list = []
net_list = []
loss_list = []
DL_list.append(train_loader_1)
DL_list.append(train_loader_2)
opt_list.append(optimizer_1)
opt_list.append(optimizer_2)
net_list.append(image_net_1)
net_list.append(image_net_2)
loss_list.append(train_loss_1)
loss_list.append(train_loss_2)


#set global update counts
num_epoch = 10




for epoch_idx in tqdm(range(num_epoch)):
    
    #loop through optimization of all local nets
    for source in range(num_local):
        
        #update local net
        for idx, sample in tqdm(enumerate(DL_list[source])):
        
            #zero gradient for local  net
            opt_list[source].zero_grad()
            
            #calc gt image
            gt_img = adjoint_op(sample['gt_nonzero_ksp'], torch.ones((1,1,sample['gt_nonzero_ksp'].shape[-1])), sample['s_maps_cplx'] ) 
            
            #calc adjoint of undersampled signal
            x_adj = adjoint_op(sample['ksp_inner'][:,0,...],sample['mask'][:,None,...],sample['s_maps_cplx'])
            
            #obtain estmiated image from network
            img_est = net_list[source](x_adj)
            #print("gt image shape:", gt_img.shape)
            #print("img_est shape:", img_est.shape)            
            #calculate loss
            loss = loss_fun(gt_img,img_est) 
        
            # Backprop
            loss.backward()
            # Verbose
            print('Epoch %d, Step %d, Batch loss Net %d :  %.4f.' % (epoch_idx, idx, source  ,loss.item()))
        
            # Optimizer Step
            opt_list[source].step()   
            
            #variable storage
            loss_list[source].append(loss.item)

    #save network params
    last_weights_1 = global_dir +'/ckpt_net1_epoch%d.pt' % epoch_idx
    last_weights_2 = global_dir +'/ckpt_net2_epoch%d.pt' % epoch_idx
    torch.save({
                    'epoch': epoch_idx,
                    'model_1_state_dict': net_list[0].state_dict(),
                    'optimizer_state_dict': opt_list[0].state_dict(),
                    'loss_1': loss_list[0]}, last_weights_1)    
    
    torch.save({
                    'epoch': epoch_idx,
                    'model_2_state_dict': net_list[1].state_dict(),
                    'optimizer_state_dict': opt_list[1].state_dict(),
                    'loss_2': loss_list[1]}, last_weights_2)  
        
        
    #share all network weights case
    if hparams.share_mode =='Full_avg':
        #average network weights
        sd_list = []
        
        for i in range(num_local):
            sd_list.append(net_list[i].state_dict())

        # Average all parameters for global network
        for key in sd_list[0]:
            
            for i in range(num_local-1):
                sd_list[0][key] = (sd_list[0][key] + sd_list[i+1][key]) 

            sd_list[0][key] = sd_list[0][key]/num_local
        
       
        for i in range(num_local):
            net_list[i].load_state_dict(sd_list[0])
        
        
    #set weights in local networks for split networks
    elif hparams.share_mode =='Split_avg':
        #average network weights
        sd_list = []
        
        for i in range(num_local):
            sd_list.append(net_list[i].res1.state_dict())

        # Average all parameters for global network
        for key in sd_list[0]:
            
            for i in range(num_local-1):
                sd_list[0][key] = (sd_list[0][key] + sd_list[i+1][key]) 

            sd_list[0][key] = sd_list[0][key]/num_local

        
        for i in range(num_local):
            net_list[i].res1.load_state_dict(sd_list[0])
    
    #elif hparams.share_mode == 'Seperate':
        #do nothing with networks
        

    last_weights_global = global_dir +'/ckpt_globalnet_epoch%d.pt' % epoch_idx
    torch.save({
                    'epoch': epoch_idx,
                    'global_state_dict': net_list[0].state_dict()}, last_weights_global)
