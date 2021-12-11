#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import torch, os, glob, copy
import numpy as np
from sys import getsizeof
from optparse import OptionParser
import argparse
import h5py


def site_loader(site_num, train_pat_count, val_pat_count):
    #INPUT:
    # sites_num = integer site number to load
    # train_pat_count = # of patients to load from each site
    # val_pat_count = # of patients to load from each site
    #OUTPUTt:
    #train_files = training kspace
    #train_maps = training maps
    #val_files = validation kspace
    #val_maps = validation maps
    ########## Training files ##############
    train_ksp_dir_1    = '/csiNAS/brett/sites_new/Training/site1/ksp' #PDFS 3T
    train_maps_dir_1    = '/csiNAS/brett/sites_new/Training/site1/maps'

    train_ksp_dir_2    = '/csiNAS/brett/sites_new/Training/site2/ksp' #PDFS 1.5T
    train_maps_dir_2    = '/csiNAS/brett/sites_new/Training/site2/maps'

    train_ksp_dir_3    = '/csiNAS/brett/sites_new/Training/site3/ksp' #PD 3T
    train_maps_dir_3    = '/csiNAS/brett/sites_new/Training/site3/maps'

    train_ksp_dir_4    = '/csiNAS/brett/sites_new/Training/site4/ksp'#PD 1.5T
    train_maps_dir_4    = '/csiNAS/brett/sites_new/Training/site4/maps'

    train_ksp_dir_5    = '/csiNAS/brett/sites_new/Training/site5/ksp' #T2 3T
    train_maps_dir_5    = '/csiNAS/brett/sites_new/Training/site5/maps'

    train_ksp_dir_6    = '/csiNAS/brett/sites_new/Training/site6/ksp' #T2 1.5T
    train_maps_dir_6    = '/csiNAS/brett/sites_new/Training/site6/maps'

    train_ksp_dir_7    = '/csiNAS/brett/sites_new/Training/site7/ksp' #FLAIR 3T
    train_maps_dir_7    = '/csiNAS/brett/sites_new/Training/site7/maps'

    train_ksp_dir_8    = '/csiNAS/brett/sites_new/Training/site8/ksp' #FLAIR 1.5T
    train_maps_dir_8    = '/csiNAS/brett/sites_new/Training/site8/maps'

    train_ksp_dir_9    = '/csiNAS/brett/sites_new/Training/site9/ksp' #POST-T1 Contrast Enhancemnt 3T
    train_maps_dir_9    = '/csiNAS/brett/sites_new/Training/site9/maps'

    train_ksp_dir_10   = '/csiNAS/brett/sites_new/Training/site10/ksp' #POST-T1 Contrast Enhancemnt 1.5T
    train_maps_dir_10   = '/csiNAS/brett/sites_new/Training/site10/maps'

    train_ksp_dir_11   = '/csiNAS/brett/sites_new/Training/site11/ksp' #T1 3.0T
    train_maps_dir_11   = '/csiNAS/brett/sites_new/Training/site11/maps'

    train_ksp_dir_12   = '/csiNAS/brett/sites_new/Training/site12/ksp' #T1 1.5T
    train_maps_dir_12   = '/csiNAS/brett/sites_new/Training/site12/maps'


    val_ksp_dir_1    = '/csiNAS/brett/sites_new/Validation/site1/ksp'
    val_maps_dir_1    = '/csiNAS/brett/sites_new/Validation/site1/maps'

    val_ksp_dir_2    = '/csiNAS/brett/sites_new/Validation/site2/ksp'
    val_maps_dir_2    = '/csiNAS/brett/sites_new/Validation/site2/maps'

    val_ksp_dir_3    = '/csiNAS/brett/sites_new/Validation/site3/ksp'
    val_maps_dir_3    = '/csiNAS/brett/sites_new/Validation/site3/maps'

    val_ksp_dir_4    = '/csiNAS/brett/sites_new/Validation/site4/ksp'
    val_maps_dir_4    = '/csiNAS/brett/sites_new/Validation/site4/maps'

    val_ksp_dir_5    = '/csiNAS/brett/sites_new/Validation/site5/ksp'
    val_maps_dir_5    = '/csiNAS/brett/sites_new/Validation/site5/maps'

    val_ksp_dir_6    = '/csiNAS/brett/sites_new/Validation/site6/ksp'
    val_maps_dir_6    = '/csiNAS/brett/sites_new/Validation/site6/maps'

    val_ksp_dir_7    = '/csiNAS/brett/sites_new/Validation/site7/ksp'
    val_maps_dir_7    = '/csiNAS/brett/sites_new/Validation/site7/maps'

    val_ksp_dir_8    = '/csiNAS/brett/sites_new/Validation/site8/ksp'
    val_maps_dir_8    = '/csiNAS/brett/sites_new/Validation/site8/maps'

    val_ksp_dir_9    = '/csiNAS/brett/sites_new/Validation/site9/ksp'
    val_maps_dir_9    = '/csiNAS/brett/sites_new/Validation/site9/maps'

    val_ksp_dir_10    = '/csiNAS/brett/sites_new/Validation/site10/ksp'
    val_maps_dir_10   = '/csiNAS/brett/sites_new/Validation/site10/maps'

    val_ksp_dir_11    = '/csiNAS/brett/sites_new/Validation/site11/ksp'
    val_maps_dir_11   = '/csiNAS/brett/sites_new/Validation/site11/maps'

    val_ksp_dir_12    = '/csiNAS/brett/sites_new/Validation/site12/ksp'
    val_maps_dir_12   = '/csiNAS/brett/sites_new/Validation/site12/maps'


    train_files_1 = sorted(glob.glob(train_ksp_dir_1 + '/*.h5'))
    train_maps_1  = sorted(glob.glob(train_maps_dir_1 + '/*.h5'))

    train_files_2 = sorted(glob.glob(train_ksp_dir_2 + '/*.h5'))
    train_maps_2  = sorted(glob.glob(train_maps_dir_2 + '/*.h5'))

    train_files_3 = sorted(glob.glob(train_ksp_dir_3 + '/*.h5'))
    train_maps_3  = sorted(glob.glob(train_maps_dir_3 + '/*.h5'))

    train_files_4 = sorted(glob.glob(train_ksp_dir_4 + '/*.h5'))
    train_maps_4  = sorted(glob.glob(train_maps_dir_4 + '/*.h5'))

    train_files_5 = sorted(glob.glob(train_ksp_dir_5 + '/*.h5'))
    train_maps_5  = sorted(glob.glob(train_maps_dir_5 + '/*.h5'))

    train_files_6 = sorted(glob.glob(train_ksp_dir_6 + '/*.h5'))
    train_maps_6  = sorted(glob.glob(train_maps_dir_6 + '/*.h5'))

    train_files_7 = sorted(glob.glob(train_ksp_dir_7 + '/*.h5'))
    train_maps_7  = sorted(glob.glob(train_maps_dir_7 + '/*.h5'))

    train_files_8 = sorted(glob.glob(train_ksp_dir_8 + '/*.h5'))
    train_maps_8  = sorted(glob.glob(train_maps_dir_8 + '/*.h5'))

    train_files_9 = sorted(glob.glob(train_ksp_dir_9 + '/*.h5'))
    train_maps_9  = sorted(glob.glob(train_maps_dir_9 + '/*.h5'))

    train_files_10 = sorted(glob.glob(train_ksp_dir_10 + '/*.h5'))
    train_maps_10  = sorted(glob.glob(train_maps_dir_10 + '/*.h5'))

    train_files_11 = sorted(glob.glob(train_ksp_dir_11 + '/*.h5'))
    train_maps_11  = sorted(glob.glob(train_maps_dir_11 + '/*.h5'))

    train_files_12 = sorted(glob.glob(train_ksp_dir_12 + '/*.h5'))
    train_maps_12  = sorted(glob.glob(train_maps_dir_12 + '/*.h5'))


    val_files_1 = sorted(glob.glob(val_ksp_dir_1 + '/*.h5'))
    val_maps_1  = sorted(glob.glob(val_maps_dir_1 + '/*.h5'))

    val_files_2 = sorted(glob.glob(val_ksp_dir_2 + '/*.h5'))
    val_maps_2  = sorted(glob.glob(val_maps_dir_2 + '/*.h5'))

    val_files_3 = sorted(glob.glob(val_ksp_dir_3 + '/*.h5'))
    val_maps_3  = sorted(glob.glob(val_maps_dir_3 + '/*.h5'))

    val_files_4 = sorted(glob.glob(val_ksp_dir_4 + '/*.h5'))
    val_maps_4  = sorted(glob.glob(val_maps_dir_4 + '/*.h5'))

    val_files_5 = sorted(glob.glob(val_ksp_dir_5 + '/*.h5'))
    val_maps_5  = sorted(glob.glob(val_maps_dir_5 + '/*.h5'))

    val_files_6 = sorted(glob.glob(val_ksp_dir_6 + '/*.h5'))
    val_maps_6  = sorted(glob.glob(val_maps_dir_6 + '/*.h5'))

    val_files_7 = sorted(glob.glob(val_ksp_dir_7 + '/*.h5'))
    val_maps_7  = sorted(glob.glob(val_maps_dir_7 + '/*.h5'))

    val_files_8 = sorted(glob.glob(val_ksp_dir_8 + '/*.h5'))
    val_maps_8  = sorted(glob.glob(val_maps_dir_8 + '/*.h5'))

    val_files_9 = sorted(glob.glob(val_ksp_dir_9 + '/*.h5'))
    val_maps_9  = sorted(glob.glob(val_maps_dir_9 + '/*.h5'))

    val_files_10 = sorted(glob.glob(val_ksp_dir_10 + '/*.h5'))
    val_maps_10  = sorted(glob.glob(val_maps_dir_10 + '/*.h5'))

    val_files_11 = sorted(glob.glob(val_ksp_dir_11 + '/*.h5'))
    val_maps_11  = sorted(glob.glob(val_maps_dir_11 + '/*.h5'))

    val_files_12 = sorted(glob.glob(val_ksp_dir_12 + '/*.h5'))
    val_maps_12  = sorted(glob.glob(val_maps_dir_12 + '/*.h5'))

    train_files = []
    train_maps  = []
    val_files   = []
    val_maps    = []
    #####################edit training and validation sizes#########################
    train_files_1_ = [train_files_1[idx] for idx in range(min(len(train_files_1),train_pat_count))]
    train_maps_1_ = [train_maps_1[idx] for idx in range(min(len(train_maps_1),train_pat_count))]
    train_files.append(train_files_1_)
    train_maps.append(train_maps_1_)
    train_files_2_ = [train_files_2[idx] for idx in range(min(len(train_files_2),train_pat_count))]
    train_maps_2_ = [train_maps_2[idx] for idx in range(min(len(train_maps_2),train_pat_count))]
    train_files.append(train_files_2_)
    train_maps.append(train_maps_2_)
    train_files_3_ = [train_files_3[idx] for idx in range(min(len(train_files_3),train_pat_count))]
    train_maps_3_ = [train_maps_3[idx] for idx in range(min(len(train_maps_3),train_pat_count))]
    train_files.append(train_files_3_)
    train_maps.append(train_maps_3_)
    train_files_4_ = [train_files_4[idx] for idx in range(min(len(train_files_4),train_pat_count))]
    train_maps_4_ = [train_maps_4[idx] for idx in range(min(len(train_maps_4),train_pat_count))]
    train_files.append(train_files_4_)
    train_maps.append(train_maps_4_)
    train_files_5_ = [train_files_5[idx] for idx in range(min(len(train_files_5),train_pat_count))]
    train_maps_5_ = [train_maps_5[idx] for idx in range(min(len(train_maps_5),train_pat_count))]
    train_files.append(train_files_5_)
    train_maps.append(train_maps_5_)
    train_files_6_ = [train_files_6[idx] for idx in range(min(len(train_files_6),train_pat_count))]
    train_maps_6_ = [train_maps_6[idx] for idx in range(min(len(train_maps_6),train_pat_count))]
    train_files.append(train_files_6_)
    train_maps.append(train_maps_6_)
    train_files_7_ = [train_files_7[idx] for idx in range(min(len(train_files_7),train_pat_count))]
    train_maps_7_ = [train_maps_7[idx] for idx in range(min(len(train_maps_7),train_pat_count))]
    train_files.append(train_files_7_)
    train_maps.append(train_maps_7_)
    train_files_8_ = [train_files_8[idx] for idx in range(min(len(train_files_8),train_pat_count))]
    train_maps_8_ = [train_maps_8[idx] for idx in range(min(len(train_maps_8),train_pat_count))]
    train_files.append(train_files_8_)
    train_maps.append(train_maps_8_)
    train_files_9_ = [train_files_9[idx] for idx in range(min(len(train_files_9),train_pat_count))]
    train_maps_9_ = [train_maps_9[idx] for idx in range(min(len(train_maps_9),train_pat_count))]
    train_files.append(train_files_9_)
    train_maps.append(train_maps_9_)
    train_files_10_ = [train_files_10[idx] for idx in range(min(len(train_files_10),train_pat_count))]
    train_maps_10_ = [train_maps_10[idx] for idx in range(min(len(train_maps_10),train_pat_count))]
    train_files.append(train_files_10_)
    train_maps.append(train_maps_10_)
    train_files_11_ = [train_files_11[idx] for idx in range(min(len(train_files_11),train_pat_count))]
    train_maps_11_ = [train_maps_11[idx] for idx in range(min(len(train_maps_11),train_pat_count))]
    train_files.append(train_files_11_)
    train_maps.append(train_maps_11_)
    train_files_12_ = [train_files_12[idx] for idx in range(min(len(train_files_12),train_pat_count))]
    train_maps_12_ = [train_maps_12[idx] for idx in range(min(len(train_maps_12),train_pat_count))]
    train_files.append(train_files_12_)
    train_maps.append(train_maps_12_)




    val_files_1_ = [val_files_1[idx] for idx in range(min(len(val_files_1),val_pat_count))]
    val_maps_1_ = [val_maps_1[idx] for idx in range(min(len(val_maps_1),val_pat_count))]
    val_files.append(val_files_1_)
    val_maps.append(val_maps_1_)
    val_files_2_ = [val_files_2[idx] for idx in range(min(len(val_files_2),val_pat_count))]
    val_maps_2_ = [val_maps_2[idx] for idx in range(min(len(val_maps_2),val_pat_count))]
    val_files.append(val_files_2_)
    val_maps.append(val_maps_2_)
    val_files_3_ = [val_files_3[idx] for idx in range(min(len(val_files_3),val_pat_count))]
    val_maps_3_ = [val_maps_3[idx] for idx in range(min(len(val_maps_3),val_pat_count))]
    val_files.append(val_files_3_)
    val_maps.append(val_maps_3_)
    val_files_4_ = [val_files_4[idx] for idx in range(min(len(val_files_4),val_pat_count))]
    val_maps_4_ = [val_maps_4[idx] for idx in range(min(len(val_maps_4),val_pat_count))]
    val_files.append(val_files_4_)
    val_maps.append(val_maps_4_)
    val_files_5_ = [val_files_5[idx] for idx in range(min(len(val_files_5),val_pat_count))]
    val_maps_5_ = [val_maps_5[idx] for idx in range(min(len(val_maps_5),val_pat_count))]
    val_files.append(val_files_5_)
    val_maps.append(val_maps_5_)
    val_files_6_ = [val_files_6[idx] for idx in range(min(len(val_files_6),val_pat_count))]
    val_maps_6_ = [val_maps_6[idx] for idx in range(min(len(val_maps_6),val_pat_count))]
    val_files.append(val_files_6_)
    val_maps.append(val_maps_6_)
    val_files_7_ = [val_files_7[idx] for idx in range(min(len(val_files_7),val_pat_count))]
    val_maps_7_ = [val_maps_7[idx] for idx in range(min(len(val_maps_7),val_pat_count))]
    val_files.append(val_files_7_)
    val_maps.append(val_maps_7_)
    val_files_8_ = [val_files_8[idx] for idx in range(min(len(val_files_8),val_pat_count))]
    val_maps_8_ = [val_maps_8[idx] for idx in range(min(len(val_maps_8),val_pat_count))]
    val_files.append(val_files_8_)
    val_maps.append(val_maps_8_)
    val_files_9_ = [val_files_9[idx] for idx in range(min(len(val_files_9),val_pat_count))]
    val_maps_9_ = [val_maps_9[idx] for idx in range(min(len(val_maps_9),val_pat_count))]
    val_files.append(val_files_9_)
    val_maps.append(val_maps_9_)
    val_files_10_ = [val_files_10[idx] for idx in range(min(len(val_files_10),val_pat_count))]
    val_maps_10_ = [val_maps_10[idx] for idx in range(min(len(val_maps_10),val_pat_count))]
    val_files.append(val_files_10_)
    val_maps.append(val_maps_10_)
    val_files_11_ = [val_files_11[idx] for idx in range(min(len(val_files_11),val_pat_count))]
    val_maps_11_ = [val_maps_11[idx] for idx in range(min(len(val_maps_11),val_pat_count))]
    val_files.append(val_files_11_)
    val_maps.append(val_maps_11_)
    val_files_12_ = [val_files_12[idx] for idx in range(min(len(val_files_12),val_pat_count))]
    val_maps_12_ = [val_maps_12[idx] for idx in range(min(len(val_maps_12),val_pat_count))]
    val_files.append(val_files_12_)
    val_maps.append(val_maps_12_)

    # train_files_out = []
    # train_maps_out = []
    # val_files_out = []
    # val_maps_out = []

    # for site in site_num:
    #     train_files_out.append(train_files[site])
    #     train_maps_out.append(train_maps[site])
    #     val_files_out.append(val_files[site])
    #     val_maps_out.append(val_maps[site])
    #############################################################################
    return train_files[site_num], train_maps[site_num], val_files[site_num], val_maps[site_num]
