#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np

def site_loader(site_num, train_pat_count, val_pat_count):
    ## INPUTS:
    # sites_num = integer site number to load
    # train_pat_count = # of patients to load from each site
    # val_pat_count = # of patients to load from each site
    ## OUTPUTS:
    # train_files = training kspace
    # train_maps = training maps
    # val_files = validation kspace
    # val_maps = validation maps

    ## Legend of training sites
    # 1 - Knee PDFS 3T
    # 2 - Knee PDFS 1.5T
    # 3 - Knee PDNFS 3T
    # 4 - Knee PDNFS 1.5T
    # 5 - Brain T2 3T
    # 6 - Brain T2 1.5T
    # 7 - Brain FLAIR 3T
    # 8 - Brain FLAIR 1.5T
    # 9 - Brain Post-T1 CE 3T
    # 10 - Brain Post-T1 CE 1.5T
    # 11 - Brain T1 3T
    # 12 - Brain T1 1.5T

    # Core directories
    core_train_dir = '/csiNAS/brett/fed_sites_new/Training'
    core_val_dir   = '/csiNAS/brett/fed_sites_new/Validation'

    # Site directories
    train_ksp_dir  = os.path.join(core_train_dir, 'site%d/ksp' % site_num)
    train_maps_dir = os.path.join(core_train_dir, 'site%d/maps' % site_num)

    val_ksp_dir  = os.path.join(core_val_dir, 'site%d/ksp' % site_num)
    val_maps_dir = os.path.join(core_val_dir, 'site%d/maps' % site_num)

    # Enumerate files
    train_files = sorted(glob.glob(train_ksp_dir + '/*.h5'))
    train_maps  = sorted(glob.glob(train_maps_dir + '/*.h5'))

    val_files = sorted(glob.glob(val_ksp_dir + '/*.h5'))
    val_maps  = sorted(glob.glob(val_maps_dir + '/*.h5'))

    train_files_out, train_maps_out = [], []
    val_files_out, val_maps_out     = [], []

    # Sub-select number of patients
    # !!! This is now random with fixed seed
    np.random.seed(2021)
    train_perm = np.random.permutation(len(train_files))
    np.random.seed(2021)
    val_perm   = np.random.permutation(len(val_files))

    train_files_out = [train_files[train_perm[idx]] for
        idx in range(min(len(train_files),train_pat_count))]
    train_maps_out  = [train_maps[train_perm[idx]] for
        idx in range(min(len(train_maps),train_pat_count))]

    val_files_out = [val_files[val_perm[idx]] for
        idx in range(min(len(val_files),val_pat_count))]
    val_maps_out  = [val_maps[val_perm[idx]] for
        idx in range(min(len(val_maps),val_pat_count))]
 
    return train_files_out, train_maps_out, val_files_out, val_maps_out