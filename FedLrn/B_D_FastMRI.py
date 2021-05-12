#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:05:56 2021

@author: brettlevac
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainFastMRI_Denoising(Dataset):
    def __init__(self, scan_list, stdev = 0.0001):
        self.scan_list = scan_list
        self.stdev = stdev
 
    def __len__(self):
        return len(self.scan_list)//2

    def __getitem__(self, idx):
        
        with h5py.File(self.scan_list[idx], 'r') as contents:
            # Get k-space for specific slice
            gt_img = np.asarray(contents['mvue'])
            noise_img = gt_img + self.stdev * (np.random.randn(gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]) + 1j * np.random.randn(gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]))
            noise_img = noise_img.astype(np.complex64)
            gt_img = gt_img[0:5,...]
            noise_img = noise_img[0:5,...]
        sample = {"gt_img": torch.tensor(gt_img), "noise_img": torch.tensor(noise_img)}
       
        return sample