#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import torch, os, glob, copy
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

from datagen import MCFullFastMRI, MCFullFastMRIBrain, crop
from models_c import MoDLDoubleUnroll
from losses import SSIMLoss, MCLoss, NMSELoss
from utils import ifft

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

from sys import getsizeof

from optparse import OptionParser
import argparse

import h5py
import xml.etree.ElementTree as ET
from mri_data import et_query

#Steps for Users who only want to increase the number of sites. Change the Following:
#1) change "num_sites" to reflect new number of sites
#2) add directories for ksp and sensitivity maps for new num_sites
#3) add dataloader for new sites
#4) add to "train_datasets" lists
num_pats = [1, 2 , 10, 20, 40 , 60]
# n_stdev =  [0.05, 0.10, 0.04, 0.09, 0.00, 0.08, 0.01, 0.07, 0.03, 0.06, 0.02]
n_stdev =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Unsupervised = False

####################USER Inputs#############################################
num_sites = 11

########## Training files for Knee ##############
core_dir_1    = '/csiNAS/mridata/fastmri_knee/multicoil_train/'
maps_dir_1    = '/csiNAS/mridata/fastmri_knee/multicoil_train_Wc0_Espirit_maps/'

train_files_1 = sorted(glob.glob(core_dir_1 + '/*.h5'))
train_maps_1  = sorted(glob.glob(maps_dir_1 + '/*.h5'))
print('Full Knee:', len(train_files_1))
# print(len(train_maps_1))

train_knee_PDFS_3T = []
train_knee_PDFS_1T = []
train_maps_knee_PDFS_3T = []
train_maps_knee_PDFS_1T = []

train_knee_PD_3T = []
train_knee_PD_1T = []
train_maps_knee_PD_3T = []
train_maps_knee_PD_1T = []

for sample_index in range(len(train_files_1)):
    with h5py.File(train_files_1[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='CORPDFS_FBK') :
            train_knee_PDFS_3T.append(train_files_1[sample_index])
            train_maps_knee_PDFS_3T.append(train_maps_1[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='CORPDFS_FBK') :
            train_knee_PDFS_1T.append(train_files_1[sample_index])
            train_maps_knee_PDFS_1T.append(train_maps_1[sample_index])
print('Knee PDFS 3T:', len(train_knee_PDFS_3T))
print('Knee PDFS 1.5T:', len(train_knee_PDFS_1T))

for sample_index in range(len(train_files_1)):
    with h5py.File(train_files_1[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='CORPD_FBK') :
            train_knee_PD_3T.append(train_files_1[sample_index])
            train_maps_knee_PD_3T.append(train_maps_1[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='CORPD_FBK') :
            train_knee_PD_1T.append(train_files_1[sample_index])
            train_maps_knee_PD_1T.append(train_maps_1[sample_index])
print('Knee PD 3T:', len(train_knee_PD_3T))
print('Knee PD 1.5T:', len(train_knee_PD_1T))

####training files for Brain###############################
core_dir_2    = '/csiNAS/mridata/fastmri_brain/multicoil_train/'
maps_dir_2    = '/csiNAS/mridata/fastmri_brain/multicoil_train_espiritWc0_mvue_ALL/'

train_files_2 = sorted(glob.glob(core_dir_2 + '/*.h5'))
train_maps_2  = sorted(glob.glob(maps_dir_2 + '/*.h5'))
print('Brain Full:', len(train_files_2))
# print(len(train_maps_2))

train_brain_T2_3T = []
train_brain_T2_1T = []
train_maps_brain_T2_3T = []
train_maps_brain_T2_1T = []

train_brain_FLAIR_3T = []
train_brain_FLAIR_1T = []
train_maps_brain_FLAIR_3T = []
train_maps_brain_FLAIR_1T = []

train_brain_POST_3T = []
train_brain_POST_1T = []
train_maps_brain_POST_3T = []
train_maps_brain_POST_1T = []

train_brain_PRECON_3T = []
train_brain_PRECON_1T = []
train_maps_brain_PRECON_3T = []
train_maps_brain_PRECON_1T = []

for sample_index in range(len(train_files_2)):
    with h5py.File(train_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT2') :
            train_brain_T2_3T.append(train_files_2[sample_index])
            train_maps_brain_T2_3T.append(train_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT2') :
            train_brain_T2_1T.append(train_files_2[sample_index])
            train_maps_brain_T2_1T.append(train_maps_2[sample_index])
print('Brain T2 3T:', len(train_brain_T2_3T))
print('Brain T2 1.5T:', len(train_brain_T2_1T))

for sample_index in range(len(train_files_2)):
    with h5py.File(train_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXFLAIR') :
            train_brain_FLAIR_3T.append(train_files_2[sample_index])
            train_maps_brain_FLAIR_3T.append(train_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXFLAIR') :
            train_brain_FLAIR_1T.append(train_files_2[sample_index])
            train_maps_brain_FLAIR_1T.append(train_maps_2[sample_index])
print('Brain FLAIR 3T:', len(train_brain_FLAIR_3T))
print('Brain FLAIR 1.5T:', len(train_brain_FLAIR_1T))


for sample_index in range(len(train_files_2)):
    with h5py.File(train_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT1POST') :
            train_brain_POST_3T.append(train_files_2[sample_index])
            train_maps_brain_POST_3T.append(train_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT1POST') :
            train_brain_POST_1T.append(train_files_2[sample_index])
            train_maps_brain_POST_1T.append(train_maps_2[sample_index])
print('Brain POST T1 3T:', len(train_brain_POST_3T))
print('Brain POST T1 1.5T:', len(train_brain_POST_1T))

for sample_index in range(len(train_files_2)):
    with h5py.File(train_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT1PRE') :
            train_brain_PRECON_3T.append(train_files_2[sample_index])
            train_maps_brain_PRECON_3T.append(train_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT1PRE') :
            train_brain_PRECON_1T.append(train_files_2[sample_index])
            train_maps_brain_PRECON_1T.append(train_maps_2[sample_index])

print('Brain PRECON 3T:', len(train_brain_PRECON_3T))
print('Brain PRECON 1.5T:', len(train_brain_PRECON_1T))
################################################################################







#Validation Files for all datasets
core_dir_1    = '/csiNAS/mridata/fastmri_knee/multicoil_val/'
maps_dir_1    = '/csiNAS/mridata/fastmri_knee/multicoil_val_Wc0_Espirit_maps/'

val_files_1 = sorted(glob.glob(core_dir_1 + '/*.h5'))
val_maps_1  = sorted(glob.glob(maps_dir_1 + '/*.h5'))
print('Full Knee Val:', len(val_files_1))
# print(len(val_maps_1))

val_knee_PDFS_3T = []
val_knee_PDFS_1T = []
val_maps_knee_PDFS_3T = []
val_maps_knee_PDFS_1T = []

val_knee_PD_3T = []
val_knee_PD_1T = []
val_maps_knee_PD_3T = []
val_maps_knee_PD_1T = []

for sample_index in range(len(val_files_1)):
    with h5py.File(val_files_1[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='CORPDFS_FBK') :
            val_knee_PDFS_3T.append(val_files_1[sample_index])
            val_maps_knee_PDFS_3T.append(val_maps_1[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='CORPDFS_FBK') :
            val_knee_PDFS_1T.append(val_files_1[sample_index])
            val_maps_knee_PDFS_1T.append(val_maps_1[sample_index])
print('Knee PDFS 3T:', len(val_knee_PDFS_3T))
print('Knee PDFS 1.5T:', len(val_knee_PDFS_1T))

for sample_index in range(len(val_files_1)):
    with h5py.File(val_files_1[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='CORPD_FBK') :
            val_knee_PD_3T.append(val_files_1[sample_index])
            val_maps_knee_PD_3T.append(val_maps_1[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='CORPD_FBK') :
            val_knee_PD_1T.append(val_files_1[sample_index])
            val_maps_knee_PD_1T.append(val_maps_1[sample_index])
print('Knee PD 3T:', len(val_knee_PD_3T))
print('Knee PD 1.5T:', len(val_knee_PD_1T))

#############################Brain Data################################
core_dir_2    = '/csiNAS/mridata/fastmri_brain/brain_multicoil_val/multicoil_val/'
maps_dir_2    = '/csiNAS/mridata/fastmri_brain/multicoil_val_espiritWc0_mvue_ALL/'

val_files_2 = sorted(glob.glob(core_dir_2 + '/*.h5'))
val_maps_2  = sorted(glob.glob(maps_dir_2 + '/*.h5'))
print('Full Brain Val:', len(val_files_2))
# print(len(val_maps_2))

val_brain_T2_3T = []
val_brain_T2_1T = []
val_maps_brain_T2_3T = []
val_maps_brain_T2_1T = []

val_brain_FLAIR_3T = []
val_brain_FLAIR_1T = []
val_maps_brain_FLAIR_3T = []
val_maps_brain_FLAIR_1T = []

val_brain_POST_3T = []
val_brain_POST_1T = []
val_maps_brain_POST_3T = []
val_maps_brain_POST_1T = []

val_brain_PRECON_3T = []
val_brain_PRECON_1T = []
val_maps_brain_PRECON_3T = []
val_maps_brain_PRECON_1T = []

torch.multiprocessing.set_sharing_strategy('file_system')


for sample_index in range(len(val_files_2)):
    with h5py.File(val_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT2') :
            val_brain_T2_3T.append(val_files_2[sample_index])
            val_maps_brain_T2_3T.append(val_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT2') :
            val_brain_T2_1T.append(val_files_2[sample_index])
            val_maps_brain_T2_1T.append(val_maps_2[sample_index])
print('Brain T2 3T:', len(val_brain_T2_3T))
print('Brain T2 1.5T:', len(val_brain_T2_1T))

for sample_index in range(len(val_files_2)):
    with h5py.File(val_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXFLAIR') :
            val_brain_FLAIR_3T.append(val_files_2[sample_index])
            val_maps_brain_FLAIR_3T.append(val_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXFLAIR') :
            val_brain_FLAIR_1T.append(val_files_2[sample_index])
            val_maps_brain_FLAIR_1T.append(val_maps_2[sample_index])
print('Brain FLAIR 3T:', len(val_brain_FLAIR_3T))
print('Brain FLAIR 1.5T:', len(val_brain_FLAIR_1T))

for sample_index in range(len(val_files_2)):
    with h5py.File(val_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT1POST') :
            val_brain_POST_3T.append(val_files_2[sample_index])
            val_maps_brain_POST_3T.append(val_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT1POST') :
            val_brain_POST_1T.append(val_files_2[sample_index])
            val_maps_brain_POST_1T.append(val_maps_2[sample_index])
print('Brain POST 3T:', len(val_brain_POST_3T))
print('Brain POST 1.5T:', len(val_brain_POST_1T))


for sample_index in range(len(val_files_2)):
    with h5py.File(train_files_2[sample_index], 'r') as hf:
        et_root = ET.fromstring(hf["ismrmrd_header"][()])
        field_strength = et_query(et_root, ['acquisitionSystemInformation', 'systemFieldStrength_T'])
#         print(field_strength)
        acquisition = hf.attrs['acquisition']

        if (float(field_strength) >2.0) and (acquisition=='AXT1PRE') :
            val_brain_PRECON_3T.append(val_files_2[sample_index])
            val_maps_brain_PRECON_3T.append(val_maps_2[sample_index])

        elif (float(field_strength) <= 2.0) and (acquisition=='AXT1PRE') :
            val_brain_PRECON_1T.append(val_files_2[sample_index])
            val_maps_brain_PRECON_1T.append(val_maps_2[sample_index])

print('Brain PRECON 3T:', len(val_brain_PRECON_3T))
print('Brain PRECON 1.5T:', len(val_brain_PRECON_1T))




for i in range(len(num_pats)):
    num_train_patients = num_pats[i]
    num_val_patients = 28


    #####################edit training and validation sizes#########################
    train_files_1 = [train_knee_PD_3T[idx] for idx in range(num_train_patients)]
    train_maps_1 = [train_maps_knee_PD_3T[idx] for idx in range(num_train_patients)]

    train_files_2 = [train_knee_PD_1T[idx] for idx in range(num_train_patients)]
    train_maps_2 = [train_maps_knee_PD_1T[idx] for idx in range(num_train_patients)]


    train_files_3 = [train_knee_PDFS_3T[idx] for idx in range(num_train_patients)]
    train_maps_3 = [train_maps_knee_PDFS_3T[idx] for idx in range(num_train_patients)]

    train_files_4 = [train_knee_PDFS_1T[idx] for idx in range(num_train_patients)]
    train_maps_4 = [train_maps_knee_PDFS_1T[idx] for idx in range(num_train_patients)]


    train_files_5 = [train_brain_T2_3T[idx] for idx in range(num_train_patients)]
    train_maps_5 = [train_maps_brain_T2_3T[idx] for idx in range(num_train_patients)]

    train_files_6 = [train_brain_T2_1T[idx] for idx in range(num_train_patients)]
    train_maps_6 = [train_maps_brain_T2_1T[idx] for idx in range(num_train_patients)]

    train_files_7 = [train_brain_FLAIR_3T[idx] for idx in range(num_train_patients)]
    train_maps_7 = [train_maps_brain_FLAIR_3T[idx] for idx in range(num_train_patients)]

    train_files_8 = [train_brain_FLAIR_1T[idx] for idx in range(num_train_patients)]
    train_maps_8 = [train_maps_brain_FLAIR_1T[idx] for idx in range(num_train_patients)]

    train_files_9 = [train_brain_POST_3T[idx] for idx in range(num_train_patients)]
    train_maps_9 = [train_maps_brain_POST_3T[idx] for idx in range(num_train_patients)]

    train_files_10 = [train_brain_POST_1T[idx] for idx in range(num_train_patients)]
    train_maps_10 = [train_maps_brain_POST_1T[idx] for idx in range(num_train_patients)]

    train_files_11 = [train_brain_PRECON_3T[idx] for idx in range(num_train_patients)]
    train_maps_11 = [train_maps_brain_PRECON_3T[idx] for idx in range(num_train_patients)]

    train_files_12 = [train_brain_PRECON_1T[idx] for idx in range(num_train_patients)]
    train_maps_12 = [train_maps_brain_PRECON_1T[idx] for idx in range(num_train_patients)]




    val_files_1 = [val_knee_PD_3T[idx] for idx in range(num_val_patients)]
    val_maps_1 = [val_maps_knee_PD_3T[idx] for idx in range(num_val_patients)]

    val_files_2 = [val_knee_PD_1T[idx] for idx in range(num_val_patients)]
    val_maps_2 = [val_maps_knee_PD_1T[idx] for idx in range(num_val_patients)]


    val_files_3 = [val_knee_PDFS_3T[idx] for idx in range(num_val_patients)]
    val_maps_3 = [val_maps_knee_PDFS_3T[idx] for idx in range(num_val_patients)]

    val_files_4 = [val_knee_PDFS_1T[idx] for idx in range(num_val_patients)]
    val_maps_4 = [val_maps_knee_PDFS_1T[idx] for idx in range(num_val_patients)]


    val_files_5 = [val_brain_T2_3T[idx] for idx in range(num_val_patients)]
    val_maps_5 = [val_maps_brain_T2_3T[idx] for idx in range(num_val_patients)]

    val_files_6 = [val_brain_T2_1T[idx] for idx in range(num_val_patients)]
    val_maps_6 = [val_maps_brain_T2_1T[idx] for idx in range(num_val_patients)]

    val_files_7 = [val_brain_FLAIR_3T[idx] for idx in range(num_val_patients)]
    val_maps_7 = [val_maps_brain_FLAIR_3T[idx] for idx in range(num_val_patients)]

    val_files_8 = [val_brain_FLAIR_1T[idx] for idx in range(num_val_patients)]
    val_maps_8 = [val_maps_brain_FLAIR_1T[idx] for idx in range(num_val_patients)]

    val_files_9 = [val_brain_POST_3T[idx] for idx in range(num_val_patients)]
    val_maps_9 = [val_maps_brain_POST_3T[idx] for idx in range(num_val_patients)]

    val_files_10 = [val_brain_POST_1T[idx] for idx in range(num_val_patients)]
    val_maps_10 = [val_maps_brain_POST_1T[idx] for idx in range(num_val_patients)]

    val_files_11 = [val_brain_PRECON_3T[idx] for idx in range(num_val_patients)]
    val_maps_11 = [val_maps_brain_PRECON_3T[idx] for idx in range(num_val_patients)]

    # val_files_12 = [val_brain_PRECON_1T[idx] for idx in range(num_val_patients)]
    # val_maps_12 = [val_maps_brain_PRECON_1T[idx] for idx in range(num_val_patients)]







    # 'num_slices' around 'central_slice' from each scan. Again this may need to be unique for each dataset
    center_slice_knee = 15 # Reasonable for fMRI
    num_slices_knee   = 5 # Around center

    center_slice_brain = 3
    num_slices_brain = 5


    #############################################################################
    plt.rcParams.update({'font.size': 12})
    plt.ioff(); plt.close('all')

    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # Fix seed
    global_seed = 1500
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    # Enable cuDNN kernel selection
    torch.backends.cudnn.benchmark = True


    ##################### Mask configs(dont touch unless doing unsupervised)#####
    train_mask_params, val_mask_params = DotMap(), DotMap()
    # 'Accel_only': accelerates in PE direction.
    # 'Gauss': Gaussian undersampling on top of PE acceleration
    train_mask_params.mask_mode       = 'Accel_only'
    train_mask_params.p_r             = 0.4 # Rho in SSDU
    train_mask_params.num_theta_masks = 1 # Split theta into a subset and apply them round-robin
    train_mask_params.theta_fraction  = 0.5 # Fraction of each mask in set, auto-adjusted to use everything
    # Validation uses all available measurements
    val_mask_params.mask_mode = 'Accel_only'
    #############################################################################



    ##################### Model config###########################################
    hparams                 = DotMap()
    hparams.mode            = 'MoDL'
    hparams.logging         = False
    hparams.mask_mode       = train_mask_params.mask_mode #US Param
    hparams.num_theta_masks = train_mask_params.num_theta_masks #US Param
    hparams.FedLrn          = True
    ############################################################################


    # Mode-specific settings
    if hparams.mode == 'MoDL':
        hparams.use_map_net     = False # No Map-net
        hparams.map_init        = 'espirit'
        hparams.block1_max_iter = 0 # Maps

    if hparams.mode == 'DeepJSense':
        hparams.use_map_net     = True
        hparams.map_init        = 'estimated' # Can be anything
        hparams.block1_max_iter = 6 # Maps
        train_maps, val_maps    = None, None
        # Map network parameters
        hparams.map_channels = 64
        hparams.map_blocks   = 4



    ##################### Image- and Map-Net parameters#########################
    hparams.img_arch     = 'ResNetSplit' # 'UNet' or 'ResNet' or 'ResNetSplit' for same split in all unrolls or 'ResUnrollSplit' to split across unrolls
    hparams.img_channels = 18 if hparams.img_arch == 'UNet' else 64
    hparams.img_blocks   = 4

    hparams.img_sep      = False # Do we use separate networks at each unroll?
    hparams.all_sep      = False  #allow all unrolls to be unique within model(True) or create just 2 networks(False): Local & Global

    hparams.downsample   = 4 # Training R
    hparams.kernel_size  = 3
    #the following parameters are only really used if using ReNetSplit for FedLrning. They specify how you want to split the ResNet
    hparams.num_blocks1     = 4
    hparams.num_blocks2     = 2
    hparams.in_channels     = 2
    hparams.latent_channels = 64
    ###########################################################################



    ######################Specify Weight Sharing Method#########################
    hparams.share_mode = 'Seperate' # either Seperate, Full_avg or Split_avg for how we want to share weights
    if hparams.img_sep == True:
        hparams.share_global = 2 #choose how many unrolls to share globally. Only necissary if we all for uni
    ##########################################################################



    # Model parametrs
    #NOTE: some of the params are used only for DeepJSense
    hparams.use_img_net        = True
    hparams.img_init           = 'estimated'
    hparams.mps_kernel_shape   = [15, 15, 9] # Always 15 coils
    hparams.l2lam_init         = 0.1
    hparams.l2lam_train        = True
    hparams.meta_unrolls_start = 1 # Starting value
    hparams.meta_unrolls_end   = 6 # Ending value - main value
    hparams.block2_max_iter    = 6 # Image
    hparams.cg_eps             = 1e-6
    hparams.verbose            = False
    # Static training parameters
    hparams.lr           = 3e-4 # Finetune if desired
    hparams.step_size    = 20    # Number of epochs to decay with gamma
    hparams.decay_gamma  = 0.5
    hparams.grad_clip    = 1. # Clip gradients
    hparams.start_epoch  = 0 # Warm start from a specific epoch
    hparams.batch_size   = 1 # !!! Unsupported !!!


    # Loss lambdas
    hparams.coil_lam = 0.
    hparams.ssim_lam = 1.

    print(n_stdev)
    num_workers = 0
    # Local training datasets
    train_dataset_1 = MCFullFastMRI(train_files_1, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=train_maps_1, mask_params=train_mask_params, noise_stdev = n_stdev[0])
    train_loader_1  = DataLoader(train_dataset_1, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_2 = MCFullFastMRI(train_files_2, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=train_maps_2, mask_params=train_mask_params, noise_stdev = n_stdev[1])
    train_loader_2  = DataLoader(train_dataset_2, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)


    train_dataset_3 = MCFullFastMRI(train_files_3, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=train_maps_3, mask_params=train_mask_params, noise_stdev = n_stdev[2])
    train_loader_3  = DataLoader(train_dataset_3, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_4 = MCFullFastMRI(train_files_4, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=train_maps_4, mask_params=train_mask_params, noise_stdev = n_stdev[3])
    train_loader_4  = DataLoader(train_dataset_4, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)


    train_dataset_5 = MCFullFastMRIBrain(train_files_5, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_5, mask_params=train_mask_params, noise_stdev = n_stdev[4])
    train_loader_5  = DataLoader(train_dataset_5, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_6 = MCFullFastMRIBrain(train_files_6, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_6, mask_params=train_mask_params, noise_stdev = n_stdev[5])
    train_loader_6  = DataLoader(train_dataset_6, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_7 = MCFullFastMRIBrain(train_files_7, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_7, mask_params=train_mask_params, noise_stdev = n_stdev[6])
    train_loader_7  = DataLoader(train_dataset_7, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_8 = MCFullFastMRIBrain(train_files_8, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_8, mask_params=train_mask_params, noise_stdev = n_stdev[7])
    train_loader_8  = DataLoader(train_dataset_8, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_9 = MCFullFastMRIBrain(train_files_9, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_9, mask_params=train_mask_params, noise_stdev = n_stdev[8])
    train_loader_9  = DataLoader(train_dataset_9, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_10 = MCFullFastMRIBrain(train_files_10, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_10, mask_params=train_mask_params, noise_stdev = n_stdev[9])
    train_loader_10  = DataLoader(train_dataset_10, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    train_dataset_11 = MCFullFastMRIBrain(train_files_11, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=train_maps_11, mask_params=train_mask_params, noise_stdev = n_stdev[10])
    train_loader_11  = DataLoader(train_dataset_11, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)







    # Local Validation datasets
    val_dataset_1 = MCFullFastMRI(val_files_1, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=val_maps_1, mask_params=val_mask_params, noise_stdev = n_stdev[0])
    val_loader_1  = DataLoader(val_dataset_1, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_2 = MCFullFastMRI(val_files_2, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=val_maps_2, mask_params=val_mask_params, noise_stdev = n_stdev[1])
    val_loader_2  = DataLoader(val_dataset_2, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)


    val_dataset_3 = MCFullFastMRI(val_files_3, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=val_maps_3, mask_params=val_mask_params, noise_stdev = n_stdev[2])
    val_loader_3  = DataLoader(val_dataset_3, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_4 = MCFullFastMRI(val_files_4, num_slices_knee, center_slice_knee,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=val_maps_4, mask_params=val_mask_params, noise_stdev = n_stdev[3])
    val_loader_4  = DataLoader(val_dataset_4, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)


    val_dataset_5 = MCFullFastMRIBrain(val_files_5, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_5, mask_params=val_mask_params, noise_stdev = n_stdev[4])
    val_loader_5  = DataLoader(val_dataset_5, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_6 = MCFullFastMRIBrain(val_files_6, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_6, mask_params=val_mask_params, noise_stdev = n_stdev[5])
    val_loader_6  = DataLoader(val_dataset_6, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_7 = MCFullFastMRIBrain(val_files_7, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_7, mask_params=val_mask_params, noise_stdev = n_stdev[6])
    val_loader_7  = DataLoader(val_dataset_7, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_8 = MCFullFastMRIBrain(val_files_8, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_8, mask_params=val_mask_params, noise_stdev = n_stdev[7])
    val_loader_8  = DataLoader(val_dataset_8, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_9 = MCFullFastMRIBrain(val_files_9, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_9, mask_params=val_mask_params, noise_stdev = n_stdev[8])
    val_loader_9  = DataLoader(val_dataset_9, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_10 = MCFullFastMRIBrain(val_files_10, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_10, mask_params=val_mask_params, noise_stdev = n_stdev[9])
    val_loader_10  = DataLoader(val_dataset_10, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)

    val_dataset_11 = MCFullFastMRIBrain(val_files_11, num_slices_brain, center_slice_brain,
                                         downsample=hparams.downsample,
                                         mps_kernel_shape=hparams.mps_kernel_shape,
                                         maps=val_maps_11, mask_params=val_mask_params, noise_stdev = n_stdev[10])
    val_loader_11  = DataLoader(val_dataset_11, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)



    # Global directory
    global_dir = 'no_noise/val_skip/Supervised/%s/%s/%s_%d_%d/latent_ch_%d/loss_ssim%.2f_coil%.2f/num_train_patients%d/seed%d' % (
        hparams.share_mode, hparams.mode, hparams.img_arch,hparams.num_blocks1,
        hparams.num_blocks2 ,hparams.latent_channels,hparams.ssim_lam,
        hparams.coil_lam, num_train_patients,global_seed )
    if not os.path.exists(global_dir):
        os.makedirs(global_dir)

    #####################Lists of dataset sites#################################
    train_datasets = []
    train_datasets.append(train_dataset_1)
    train_datasets.append(train_dataset_2)
    train_datasets.append(train_dataset_3)
    train_datasets.append(train_dataset_4)
    train_datasets.append(train_dataset_5)
    train_datasets.append(train_dataset_6)
    train_datasets.append(train_dataset_7)
    train_datasets.append(train_dataset_8)
    train_datasets.append(train_dataset_9)
    train_datasets.append(train_dataset_10)
    train_datasets.append(train_dataset_11)


    train_loader = []
    train_loader.append(train_loader_1)
    train_loader.append(train_loader_2)
    train_loader.append(train_loader_3)
    train_loader.append(train_loader_4)
    train_loader.append(train_loader_5)
    train_loader.append(train_loader_6)
    train_loader.append(train_loader_7)
    train_loader.append(train_loader_8)
    train_loader.append(train_loader_9)
    train_loader.append(train_loader_10)
    train_loader.append(train_loader_11)



    val_loaders = []
    val_loaders.append(val_loader_1)
    val_loaders.append(val_loader_2)
    val_loaders.append(val_loader_3)
    val_loaders.append(val_loader_4)
    val_loaders.append(val_loader_5)
    val_loaders.append(val_loader_6)
    val_loaders.append(val_loader_7)
    val_loaders.append(val_loader_8)
    val_loaders.append(val_loader_9)
    val_loaders.append(val_loader_10)
    val_loaders.append(val_loader_11)

    ############################################################################
    #size of validation sets
    num_val_samp = []
    num_val_samp.append(num_val_patients*num_slices_knee)
    num_val_samp.append(num_val_patients*num_slices_knee)
    num_val_samp.append(num_val_patients*num_slices_knee)
    num_val_samp.append(num_val_patients*num_slices_knee)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)
    num_val_samp.append(num_val_patients*num_slices_brain)

    #########Set up one model that each site will load states to################
    model = MoDLDoubleUnroll(hparams)
    model  = model.cuda()


    # Switch to train
    model.train()

    torch.save({
        'model': model.state_dict(),
        }, global_dir + '/Initial_weights.pt')

    ############################################################################

    # Count parameters
    total_params = np.sum([np.prod(p.shape) for p
                           in model.parameters() if p.requires_grad])
    print('Total parameters %d' % total_params)

    # Criterions
    ssim           = SSIMLoss().cuda()
    multicoil_loss = MCLoss().cuda()
    pixel_loss     = torch.nn.MSELoss(reduction='sum')
    nmse_loss      = NMSELoss()

    ############################################################################
    #######end of setting training configurations###############################
    ############################################################################



    #############Training Occurs Below(shoulnt really need to touch this)#######

    # For each number of unrolls
    for num_unrolls in range(hparams.meta_unrolls_start, hparams.meta_unrolls_end+1):

        model.train()
        # Warm-up or not
        if num_unrolls < hparams.meta_unrolls_end:
            hparams.num_epochs = 1
        else:
            hparams.num_epochs = 100


        # Get optimizer and scheduler(One for each site)
        optimizers = []
        schedulers = []
        for site in range(num_sites):
            optimizers.append(Adam(model.parameters(), lr=hparams.lr))
            schedulers.append(StepLR(optimizers[site], hparams.step_size,
                               gamma=hparams.decay_gamma))

        # If we're beyond the first step, preload weights and state
        if num_unrolls > hparams.meta_unrolls_start:
            target_dir = global_dir + '/N%d_n%d_lamInit%.3f' % (
                num_unrolls-1, hparams.block1_max_iter,
                hparams.l2lam_init)

            # Get and load previous weights from last warm up step(note untill laast unroll step is reached each is only run for one epoch)
            contents = torch.load(target_dir + '/ckpt_epoch' +str(0)+  'site'+str(0)+'after_share_weights.pt')
            # contents = torch.load(target_dir + '/best_weights.pt')

            model.load_state_dict(contents['model' + str(0) + '_state_dict'])
            for site in range(num_sites):
                contents = torch.load(target_dir + '/ckpt_epoch' +str(0)+  'site'+str(site)+'after_share_weights.pt')
                optimizers[site].load_state_dict(contents['optimizer' + str(site) + '_state_dict'])

        #create logs for each site in list format
        best_loss = []
        training_log = []
        loss_log = []
        ssim_log = []
        coil_log = []
        nmse_log = []
        running_training = []
        running_loss = []
        running_nmse = []
        running_ssim = []
        running_coil = []
        val_SSIM = []
        for i in range(num_sites):
            best_loss.append(np.inf)
            training_log.append([])
            loss_log.append([])
            ssim_log.append([])
            coil_log.append([])
            nmse_log.append([])
            running_training.append(0.)
            running_loss.append(0.)
            running_nmse.append(0.)
            running_ssim.append(-1.)
            running_coil.append(0.)
            val_SSIM.append([])



        local_dir = global_dir + '/N%d_n%d_lamInit%.3f' % (
                num_unrolls, hparams.block1_max_iter,
                hparams.l2lam_init)
        if not os.path.isdir(local_dir):
            os.makedirs(local_dir)

        # Preload from the same model hyperparameters
        # I dont currently use this loader since I always start from 0, but if I do it needs some edits
        # if hparams.start_epoch > 0:
        #     contents = torch.load(local_dir + '/ckpt_epoch%d.pt' % (hparams.start_epoch-1))
        #     for site in range(num_sites):
        #         model.load_state_dict(contents['model'+str(site)+'_state_dict'])
        #         optimizers[site].load_state_dict(contents['optimizer'+str(site)+'_state_dict'])
        #         # Increment scheduler
        #         schedulers[site].last_epoch = hparams.start_epoch-1



    # For each epoch
        for epoch_idx in range(hparams.start_epoch, hparams.num_epochs):
            # Log one divergence event per epoch
            first_time, stable_model, stable_opt = True, None, None
            # For each site
            for site in range(num_sites):

                #need this so that on the first unroll all models start with the random initial wieghts
                if num_unrolls == hparams.meta_unrolls_start :
                    temp = torch.load(global_dir + '/Initial_weights.pt')
                    model.load_state_dict(temp['model'])
                #load model for each site sequencitally at final unroll level only at training time to preserve memory
                elif (num_unrolls ==  hparams.meta_unrolls_end) & (epoch_idx>0):
                    saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx-1)+  'site'+str(site)+'after_share_weights.pt')
                    model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                    optimizers[site].load_state_dict(saved_model['optimizer' + str(site) + '_state_dict'])
                elif (num_unrolls <= hparams.meta_unrolls_end) & (epoch_idx==0):
                    target_dir = global_dir + '/N%d_n%d_lamInit%.3f' % (
                        num_unrolls-1, hparams.block1_max_iter,
                        hparams.l2lam_init)
                    # Get and load previous weights from last warm up step(note untill laast unroll step is reached each is only run for one epoch)
                    contents = torch.load(target_dir + '/ckpt_epoch' +str(0)+  'site'+str(site)+'after_share_weights.pt')
                    # contents = torch.load(target_dir + '/best_weights.pt')
                    model.load_state_dict(contents['model' + str(site) + '_state_dict'])
                    optimizers[site].load_state_dict(contents['optimizer' + str(site) + '_state_dict'])


                for sample_idx, sample in tqdm(enumerate(train_loader[site])):
                    # Move to CUDA
                    #print('Memory allocated Before sample:', torch.cuda.memory_allocated())
                    for key in sample.keys():
                        try:
                            sample[key] = sample[key].cuda()
                        except:
                            pass
                    #print('Memory allocated after sample:', torch.cuda.memory_allocated())
                    # Get outputs
                    est_img_kernel, est_map_kernel, est_ksp = \
                        model(sample, num_unrolls,
                              train_mask_params.num_theta_masks)

                    # Extra padding with zero lines - to restore resolution
                    est_ksp_padded = F.pad(est_ksp, (
                            torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
                            torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))

                    # Convert to image domain
                    est_img_coils = ifft(est_ksp_padded)

                    # RSS images
                    est_img_rss = torch.sqrt(
                        torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))

                    # Central crop
                    est_crop_rss = crop(est_img_rss, sample['ref_rss'].shape[-2],
                                        sample['ref_rss'].shape[-1])
                    gt_rss       = sample['ref_rss']
                    data_range   = sample['data_range']

                    # SSIM loss with crop
                    ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)

                    if Unsupervised ==True:
                        if hparams.mask_mode == 'Gauss':
                            coil_loss = multicoil_loss(sample['mask_outer'][None, ...] * est_ksp,
                                               sample['ksp_outer'])
                        elif hparams.mask_mode == 'Accel_only':
                            coil_loss = multicoil_loss(sample['mask'][None, ...] * est_ksp,
                                               sample['ksp'])

                        loss = coil_loss



                    elif Unsupervised == False:
                        coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])

                        loss = hparams.ssim_lam * ssim_loss + hparams.coil_lam * coil_loss


                    # Other Loss for tracking
                    with torch.no_grad():
                        pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                        nmse      = nmse_loss(gt_rss,est_crop_rss)


                    # Keep a running loss
                    running_training[site] = 0.99 * running_training[site] + 0.01 * loss.item() if running_training[site] > 0. else loss.item()
                    running_ssim[site] = 0.99 * running_ssim[site] + 0.01 * (1-ssim_loss.item()) if running_ssim[site] > -1. else (1-ssim_loss.item())
                    running_loss[site] = 0.99 * running_loss[site] + 0.01 * pix_loss.item() if running_loss[site] > 0. else pix_loss.item()
                    running_coil[site] = 0.99 * running_coil[site] + 0.01 * coil_loss.item() if running_coil[site] > 0. else coil_loss.item()
                    running_nmse[site] = 0.99 * running_nmse[site] + 0.01 * nmse.item() if running_nmse[site] > 0. else nmse.item()

                    training_log[site].append(running_training[site])
                    loss_log[site].append(running_loss[site])
                    ssim_log[site].append(running_ssim[site])
                    coil_log[site].append(running_coil[site])
                    nmse_log[site].append(running_nmse[site])
                    # Save a stable model state
                    # stable_model = copy.deepcopy(models[site].state_dict())
                    # stable_opt   = copy.deepcopy(optimizers[site].state_dict())

                    # Backprop
                    optimizers[site].zero_grad()
                    loss.backward()
                    # For MoDL (?), clip gradients
                    torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
                    optimizers[site].step()

                    # Save best model
                    #if running_loss[site] < best_loss[site]:
                        #best_loss[site] = running_loss[site]
                    torch.save({
                        'epoch': epoch_idx,
                        'sample_idx': sample_idx,
                        'model'+str(site)+'_state_dict': model.state_dict(),
                        'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                        'ssim'+str(site)+'_log': ssim_log[site],
                        'loss'+str(site)+'_log': loss_log[site],
                        'coil'+str(site)+'_log': coil_log[site],
                        'nmse'+str(site)+'_log': nmse_log[site],
                        'loss'+ str(site): loss,
                        'hparams': hparams,
                        'train_mask_params': train_mask_params}, local_dir + '/site'+str(site)+'last_weights.pt')

                   # Verbose
                    print('Epoch %d, Site %d ,Step %d, Batch loss %.4f. Avg. SSIM %.4f, Avg. RSS %.4f, Avg. Coils %.4f, Avg. NMSE %.4f' % (
                        epoch_idx, site, sample_idx, loss.item(), running_ssim[site], running_loss[site], running_coil[site], running_nmse[site]))


            ###############Do weight averaging within resnet blocks###############
            with torch.no_grad():
                if hparams.img_sep == False:
                    if hparams.share_mode == 'Full_avg':
                        print("Full_avg")
                        state_dict_accumulator = MoDLDoubleUnroll(hparams)
                        #print("device:" , state_dict_accumulator.get_device())
                        saved_model = torch.load(local_dir + '/site'+str(0)+'last_weights.pt')

                        state_dict_accumulator.load_state_dict(saved_model['model' + str(0) + '_state_dict'])
                        state_dict_accumulator = state_dict_accumulator.image_net.state_dict()

                        # state_dict_transitive.load_state_dict(saved_model['model' + str(0) + '_state_dict'])
                        # state_dict_transitive = state_dict_transitive.image_net.state_dict()

                        for key in state_dict_accumulator:
                            for site in range(num_sites-1):
                                state_dict_transitive  = MoDLDoubleUnroll(hparams)
                                temp_model = torch.load(local_dir + '/site'+str(site+1)+'last_weights.pt')
                                state_dict_transitive.load_state_dict(temp_model['model' + str(site+1) + '_state_dict'])
                                state_dict_transitive = state_dict_transitive.image_net.state_dict()

                                state_dict_accumulator[key]  = (state_dict_accumulator[key]  + state_dict_transitive[key])

                            state_dict_accumulator[key] = state_dict_accumulator[key]/num_sites

                        for site in range(num_sites):
                            model.image_net.load_state_dict(state_dict_accumulator)

                            torch.save({
                                'epoch': epoch_idx,
                                'sample_idx': sample_idx,
                                'model'+str(site)+'_state_dict': model.state_dict(),
                                'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                                'ssim'+str(site)+'_log': ssim_log[site],
                                'loss'+str(site)+'_log': loss_log[site],
                                'coil'+str(site)+'_log': coil_log[site],
                                'nmse'+str(site)+'_log': nmse_log[site],
                                'loss'+ str(site): loss,
                                'hparams': hparams,
                                'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')


                    elif hparams.share_mode == 'Split_avg':
                        print("Split_avg")
                        state_dict_accumulator = MoDLDoubleUnroll(hparams)

                        saved_model = torch.load(local_dir + '/site'+str(0)+'last_weights.pt')

                        state_dict_accumulator.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                        state_dict_accumulator = state_dict_accumulator.image_net.res1.state_dict()

                        for key in state_dict_accumulator:
                            for site in range(num_sites-1):
                                state_dict_transitive = MoDLDoubleUnroll(hparams)
                                saved_model = torch.load(local_dir + '/site'+str(site+1)+'last_weights.pt')
                                state_dict_transitive.load_state_dict(saved_model['model' + str(site+1) + '_state_dict'])
                                state_dict_transitive = state_dict_transitive.image_net.res1.state_dict()

                                state_dict_accumulator[key]  = (state_dict_accumulator[key]  + state_dict_transitive[key])

                            state_dict_accumulator[key] = state_dict_accumulator[key]/num_sites

                        for site in range(num_sites):
                            saved_model = torch.load(local_dir + '/site'+str(site)+'last_weights.pt')
                            model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                            model.image_net.res1.load_state_dict(state_dict_accumulator)

                            torch.save({
                                'epoch': epoch_idx,
                                'sample_idx': sample_idx,
                                'model'+str(site)+'_state_dict': model.state_dict(),
                                'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                                'ssim'+str(site)+'_log': ssim_log[site],
                                'loss'+str(site)+'_log': loss_log[site],
                                'coil'+str(site)+'_log': coil_log[site],
                                'nmse'+str(site)+'_log': nmse_log[site],
                                'loss'+ str(site): loss,
                                'hparams': hparams,
                                'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')

                    elif hparams.share_mode == 'Seperate':
                        print("Seperate")
                        for site in range(num_sites):
                            saved_model = torch.load(local_dir + '/site'+str(site)+'last_weights.pt')
                            model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                            torch.save({
                                'epoch': epoch_idx,
                                'sample_idx': sample_idx,
                                'model'+str(site)+'_state_dict': model.state_dict(),
                                'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                                'ssim'+str(site)+'_log': ssim_log[site],
                                'loss'+str(site)+'_log': loss_log[site],
                                'coil'+str(site)+'_log': coil_log[site],
                                'nmse'+str(site)+'_log': nmse_log[site],
                                'loss'+ str(site): loss,
                                'hparams': hparams,
                                'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')


                elif hparams.img_sep == True:
                    if hparams.share_mode == 'Split_avg':
                        print("Across unroll, Split_avg")
                        state_dict_accumulator = MoDLDoubleUnroll(hparams)

                        saved_model = torch.load(local_dir + '/site'+str(0)+'last_weights.pt')

                        state_dict_accumulator.load_state_dict(saved_model['model' + str(0) + '_state_dict'])
                        state_dict_accumulator = state_dict_accumulator.image_net[0:hparams.share_unroll].state_dict()

                        for key in state_dict_accumulator:
                            # print(key)
                            for site in range(num_sites-1):
                                state_dict_transitive = MoDLDoubleUnroll(hparams)
                                saved_model = torch.load(local_dir + '/site'+str(site+1)+'last_weights.pt')
                                state_dict_transitive.load_state_dict(saved_model['model' + str(site+1) + '_state_dict'])
                                state_dict_transitive = state_dict_transitive.image_net[0:hparams.share_unroll].state_dict()

                                state_dict_accumulator[key]  = (state_dict_accumulator[key]  + state_dict_transitive[key])

                            state_dict_accumulator[key] = state_dict_accumulator[key]/num_sites


                        for site in range(num_sites):
                            saved_model = torch.load(local_dir + '/site'+str(site)+'last_weights.pt')
                            model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                            model.image_net[0:hparams.share_unroll].load_state_dict(state_dict_accumulator)
                            # model.image_net.load_state_dict(state_dict_accumulator)
                            torch.save({
                                'epoch': epoch_idx,
                                'sample_idx': sample_idx,
                                'model'+str(site)+'_state_dict': model.state_dict(),
                                'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                                'ssim'+str(site)+'_log': ssim_log[site],
                                'loss'+str(site)+'_log': loss_log[site],
                                'coil'+str(site)+'_log': coil_log[site],
                                'nmse'+str(site)+'_log': nmse_log[site],
                                'loss'+ str(site): loss,
                                'hparams': hparams,
                                'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')





                    elif hparams.share_mode == 'Seperate':
                        print("Seperate")
                        for site in range(num_sites):
                            saved_model = torch.load(local_dir + '/site'+str(site)+'last_weights.pt')
                            model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])
                            torch.save({
                                'epoch': epoch_idx,
                                'sample_idx': sample_idx,
                                'model'+str(site)+'_state_dict': model.state_dict(),
                                'optimizer'+str(site)+'_state_dict': optimizers[site].state_dict(),
                                'ssim'+str(site)+'_log': ssim_log[site],
                                'loss'+str(site)+'_log': loss_log[site],
                                'coil'+str(site)+'_log': coil_log[site],
                                'nmse'+str(site)+'_log': nmse_log[site],
                                'loss'+ str(site): loss,
                                'hparams': hparams,
                                'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')
        #####################################################################

            ############Add weight sharing for full blocks in unrolls############
            #####################################################################

            # Save models
            # last_weights = local_dir +'/ckpt_epoch%d.pt' % epoch_idx
            # torch.save({
            #     'epoch': epoch_idx,
            #     'model0_state_dict': models[0].state_dict(),
            #     'model1_state_dict': models[1].state_dict(),
            #     'optimizer0_state_dict': optimizers[0].state_dict(),
            #     'optimizer1_state_dict': optimizers[1].state_dict(),
            #     'ssim_log': ssim_log,
            #     'loss_log': loss_log,
            #     'coil_log': coil_log,
            #     'nmse_log': nmse_log,
            #     'loss': loss,
            #     'hparams': hparams,
            #     'train_mask_params': train_mask_params}, last_weights)

            # Scheduler
            for site in range(num_sites):
                schedulers[site].step()


            if (epoch_idx%10==0) and (num_unrolls == hparams.meta_unrolls_end):


                for site in range(num_sites):
                    # After each epoch, check some validation samples
                    saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')
                    model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])

                    model.eval()
                    iterator = iter(val_loaders[site])
                    # Plot
                    plt.figure()
                    for sample_idx in range(4):
                        print("sample index:", sample_idx)
                        sample = next(iterator)
                        # Move to CUDA
                        for key, value in sample.items():
                            try:
                                sample[key] = sample[key].cuda()
                            except:
                                pass

                        # Get outputs
                        with torch.no_grad():
                            # Estimate
                            est_img_kernel, est_map_kernel, est_ksp = \
                                model(sample, num_unrolls)

                            # Extra padding with dead zones
                            est_ksp_padded = F.pad(est_ksp, (
                                torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
                                torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))

                            # Convert to image domain
                            est_img_coils = ifft(est_ksp_padded)

                            # RSS images
                            est_img_rss = torch.sqrt(torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))
                            # Central crop
                            est_crop_rss = crop(est_img_rss, sample['ref_rss'].shape[-2],
                                                sample['ref_rss'].shape[-1])
                            # Losses
                            ssim_loss = ssim(est_crop_rss[:, None], sample['ref_rss'][:, None],
                                             sample['data_range'])
                            l1_loss   = pixel_loss(est_crop_rss, sample['ref_rss'])

                        # Plot
                        plt.subplot(2, 4, sample_idx+1)
                        plt.imshow(torch.flip(sample['ref_rss'][0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=0.1, cmap='gray')
                        plt.axis('off'); plt.title('GT RSS')
                        plt.subplot(2, 4, sample_idx+1+4*1)
                        plt.imshow(torch.flip(est_crop_rss[0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=0.1, cmap='gray')
                        plt.axis('off'); plt.title('Ours - RSS')

                    # Save
                    plt.tight_layout()
                    plt.savefig(local_dir + '/val_site' + str(site) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
                    plt.close()

                    # Plot training dynamics
                    plt.figure()
                    plt.subplot(1, 3, 1); plt.semilogy(training_log[site], linewidth=2.);
                    plt.grid(); plt.xlabel('Step'); plt.title('Training loss')
                    plt.subplot(1, 3, 2); plt.semilogy(ssim_log[site], linewidth=2.);
                    plt.grid(); plt.xlabel('Step'); plt.title('Training SSIM')
                    plt.subplot(1, 3, 3); plt.semilogy(nmse_log[site], linewidth=2.);
                    plt.grid(); plt.xlabel('Step'); plt.title('Training NMSE')
                    plt.tight_layout()
                    plt.savefig(local_dir + '/train_loss_site' + str(site) + '_epoch%d.png' % epoch_idx, dpi=300)
                    plt.close()

                print("VALIDATION!!!!!!!!!!!!")
         #Full Validation metrics for each site
                for site in range(num_sites):
                    ssim_epoch_avg = 0.

                    saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(site)+'after_share_weights.pt')
                    model.load_state_dict(saved_model['model' + str(site) + '_state_dict'])

                    for sample_idx, sample in tqdm(enumerate(val_loaders[site])):
                        # Move to CUDA
                        for key, value in sample.items():
                            try:
                                sample[key] = sample[key].cuda()
                            except:
                                pass

                        with torch.no_grad():
                        # Get outputs
                            est_img_kernel, est_map_kernel, est_ksp = \
                                model(sample, num_unrolls,
                                      train_mask_params.num_theta_masks)

                            # Extra padding with zero lines - to restore resolution
                            est_ksp_padded = F.pad(est_ksp, (
                                    torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
                                    torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))

                            # Convert to image domain
                            est_img_coils = ifft(est_ksp_padded)

                            # RSS images
                            est_img_rss = torch.sqrt(
                                torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))

                            # Central crop
                            est_crop_rss = crop(est_img_rss, sample['ref_rss'].shape[-2],
                                                sample['ref_rss'].shape[-1])
                            gt_rss       = sample['ref_rss']
                            data_range   = sample['data_range']

                            # Other Loss for tracking
                            with torch.no_grad():
                                pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                                # SSIM loss with crop
                                ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)

                                coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])

                            ssim_epoch_avg = ssim_epoch_avg + (1.0-ssim_loss.item())

                    val_SSIM[site].append(ssim_epoch_avg/num_val_samp[site])


                # Plot validation dynamics
                plt.figure()
                plt.subplot(1, 4, 1); plt.semilogy(val_SSIM[0], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM K3TPD')
                plt.subplot(1, 4, 2); plt.semilogy(val_SSIM[1], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM K1.5TPD')
                plt.subplot(1, 4, 3); plt.semilogy(val_SSIM[2], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM K3TPDFS')
                plt.subplot(1, 4, 4); plt.semilogy(val_SSIM[3], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM K1.5TPDFS')


                plt.tight_layout()
                plt.savefig(local_dir + '/Val_SSIM_Knee' '_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()


                plt.figure()
                plt.subplot(1, 4, 1); plt.semilogy(val_SSIM[4], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B3TT2')
                plt.subplot(1, 4, 2); plt.semilogy(val_SSIM[5], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B1.5TT2')
                plt.subplot(1, 4, 3); plt.semilogy(val_SSIM[6], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B3TFLAIR')
                plt.subplot(1, 4, 4); plt.semilogy(val_SSIM[7], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B1.5TFLAIR')

                plt.tight_layout()
                plt.savefig(local_dir + '/Val_SSIM_Brain_1' '_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()


                plt.figure()
                plt.subplot(1, 3, 1); plt.semilogy(val_SSIM[8], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B3TPOST')
                plt.subplot(1, 3, 2); plt.semilogy(val_SSIM[9], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B1.5TPOST')
                plt.subplot(1, 3, 3); plt.semilogy(val_SSIM[10], linewidth=2.);
                plt.grid(); plt.xlabel('Epoch'); plt.title('SSIM B3TPRE')


                plt.tight_layout()
                plt.savefig(local_dir + '/Val_SSIM_Brain_2' '_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()


            model.train()
