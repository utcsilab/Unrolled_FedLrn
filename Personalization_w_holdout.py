#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import torch, os, copy
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

from datagen import MCFullFastMRI, crop
from models_c import MoDLDoubleUnroll
from losses import SSIMLoss, MCLoss, NMSELoss
from utils import ifft

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

from argparse import ArgumentParser
from site_loader import site_loader



def get_args():
    parser = ArgumentParser()
    parser.add_argument('--seed'    , type=int, default=1500     , help='random seed to use')
    parser.add_argument('--GPU'     , type=int                   , help='GPU to Use')
    parser.add_argument('--num_work', type=int                   , help='number of workers to use')
    parser.add_argument('--site', type=int             , help='site to personalize')
    parser.add_argument('--train_pats', type=int             , help='patients available for training at new site')
    parser.add_argument('--holdout_pats', type=int             , help='patients used for determining early stopping')
    parser.add_argument('--dataset', type=str, default = 'fastMRI'             , help='is the client from fastMRI or Stanford')
    parser.add_argument('--global_opt', type=str, default = 'FedAvg'             , help='global Optimizer used for file path purposes')
    parser.add_argument('--LR'      , type=float, default=3e-4   , help='learning rate for training')
    args = parser.parse_args()
    return args

# Get arguments
args = get_args()
print(args)

GPU_ID                = args.GPU
global_seed           = args.seed
num_workers           = args.num_work
site                  = args.site
num_train_pats        = args.train_pats
num_holdout_pats      = args.holdout_pats
dataset               = args.dataset
LR                    = args.LR
global_opt            = args.global_opt

plt.rcParams.update({'font.size': 12})
plt.ioff(); plt.close('all')

# Maybe
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# Always !!!
torch.backends.cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# Target weights for personalized ('one-round') baseline
if global_opt == 'FedAvg':
    target_dir = '/fast/marius/federated/results_from_ECEA01620/federated/\
    clientAdam_clients10_sites2_5_7_9_11_6_8_10_12_5/personalLam0/FedAvg/sync100/\
    UNet_pool3_ch16/train_pats_5_5_5_5_5_5_5_5_5_5/seed1500/N6_n0_lamInit0.100/'
    hparams_file = torch.load(target_dir + 'round245_client0_before_download.pt')
    target_file = target_dir + 'fed_download.pt'

elif global_opt == 'FedAdam':
    target_dir = '/fast/marius/federated/newest_results_jan14/\
    clientAdam_clients10_sites2_5_7_9_11_6_8_10_12_5/personalLam0/\
    FedAdam_tau1.0e-03_b19.0e-01_b29.9e-01_eta5.0e-02/sync100/UNet_pool3_ch16/\
    train_pats_5_5_5_5_5_5_5_5_5_5/seed1500/N6_n0_lamInit0.100/'
    hparams_file = torch.load(target_dir + 'round239_client0_before_download.pt')
    target_file = target_dir + 'fed_download.pt'

elif global_opt == 'Scaffold':
    target_dir  = '/fast/marius/federated/newest_results_jan14/clientAdam_clients10_sites1_2_4_5_6_7_8_9_10_11/personalLam0/Scaffold/clientLR6.0e-04_clientAdam/sync100/UNet_pool3_ch16/train_pats_5_5_5_5_5_5_5_5_5_5/seed1500/N6_n0_lamInit0.100/'
    hparams_file = torch.load(target_dir + 'round239_client0_before_download.pt')
    target_file = target_dir + 'fed_download.pt'


elif global_opt == 'centralized':
    target_dir  = '/fast/marius/federated/newest_results_jan14/centralized/sites_1_2_4_5_6_7_8_9_10_11/UNet_pool3_ch16/train_pats_5_5_5_5_5_5_5_5_5_5/seed1500/N6_n6_lamInit0.100/'
    hparams_file = torch.load(target_dir + 'ckpt_epoch47.pt')
    target_file = target_dir + 'ckpt_epoch47.pt'

# target_file = target_dir + 'fed_download.pt'





contents = torch.load(target_file)

# hparams_file = torch.load(target_dir + '/round245_client0_before_download.pt')
hparams  = hparams_file['hparams']




# !!!! MAKE AND LOAD STATE OF ADAM OPTIMIZER !!!!
# !!!!!

# Dataset stuff
num_val_pats       =  20
center_slice_knee  = 17
num_slices_knee    = 10
center_slice_brain = 5
num_slices_brain   = 10


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

# Criterions
ssim           = SSIMLoss().cuda()
multicoil_loss = MCLoss().cuda()
pixel_loss     = torch.nn.MSELoss(reduction='sum')
nmse_loss      = NMSELoss()


for holdout_num in range(num_train_pats):
    # Load model
    model = MoDLDoubleUnroll(hparams)
    model = model.cuda()
    model.load_state_dict(contents['model_state_dict'])
    model.train()
    if dataset == 'fastMRI':
        print('fastMRI Site')
        # Get all the filenames from the site
        all_train_files, all_train_maps, all_val_files, all_val_maps = \
            site_loader(site, num_train_pats, num_val_pats)
        #Seperate holdout samples from trianing samples
        holdout_pat_files = [all_train_files[holdout_num]]
        holdout_pat_maps  = [all_train_maps[holdout_num]]

        all_train_files.pop(holdout_num)
        all_train_maps.pop(holdout_num)
        print('training files:',all_train_files)
        print('holdout files:',holdout_pat_files)

        if site <= 4: # must be a knee site
            center_slice = center_slice_knee
            num_slices   = num_slices_knee
        elif site > 4: # must be a brain site
            center_slice = center_slice_brain
            num_slices   = num_slices_brain
        val_mask_params = DotMap()
        # !!! Deprecated, but still required
        val_mask_params.mask_mode = 'Accel_only'

        # !!!!! Make a train dataset and loader here
        # Create client dataset and loader
        train_dataset = MCFullFastMRI(all_train_files, num_slices, center_slice,
                                      downsample=hparams.downsample,
                                      mps_kernel_shape=hparams.mps_kernel_shape,
                                      maps=all_train_maps, mask_params=train_mask_params,
                                      noise_stdev = 0.0)
        train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True,
                                   pin_memory=True)

        holdout_dataset = MCFullFastMRI(holdout_pat_files, num_slices, center_slice,
                                      downsample=hparams.downsample,
                                      mps_kernel_shape=hparams.mps_kernel_shape,
                                      maps=holdout_pat_maps, mask_params=train_mask_params,
                                      noise_stdev = 0.0)
        holdout_loader  = DataLoader(holdout_dataset, batch_size=hparams.batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True,
                                   pin_memory=True)

        val_dataset = MCFullFastMRI(all_val_files, num_slices, center_slice,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    maps=all_val_maps, mask_params=val_mask_params,
                                    noise_stdev=0.0, scramble=False)
        val_loader = DataLoader(val_dataset, batch_size=1,
                     shuffle=False, num_workers=num_workers, drop_last=False,
                     pin_memory=True, prefetch_factor=1)



    if dataset == 'Stanford':
        # Get all the filenames from the site
        all_train_files = glob.glob()
        all_val_files   = glob.glob()


        center_slice = 170
        num_slices   = 50
        val_mask_params = DotMap()
        # !!! Deprecated, but still required
        val_mask_params.mask_mode = 'Accel_only'

        # !!!!! Make a train dataset and loader here
        # Create client dataset and loader
        train_dataset = MCFullStanford(all_train_files, num_slices, center_slice,
                                      downsample=hparams.downsample,
                                      mps_kernel_shape=hparams.mps_kernel_shape,
                                      mask_params=train_mask_params,
                                      noise_stdev = 0.0)
        train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True,
                                   pin_memory=True)


        val_dataset = MCFullStanford(all_val_files, num_slices, center_slice,
                                    downsample=hparams.downsample,
                                    mps_kernel_shape=hparams.mps_kernel_shape,
                                    mask_params=val_mask_params,
                                    noise_stdev=0.0, scramble=False)
        val_loader = DataLoader(val_dataset, batch_size=1,
                     shuffle=False, num_workers=num_workers, drop_last=False,
                     pin_memory=True, prefetch_factor=1)

        result_dir =  target_dir + 'personalized_stanfordsite/pats_%d/' %(num_train_pats)



    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, hparams.decay_epochs,
                       gamma=hparams.decay_gamma)




    # create logs for each client in list format
    best_loss, training_log = np.inf, []
    loss_log, ssim_log      = [], []
    coil_log, nmse_log      = [], []
    running_training        = 0.
    running_loss, running_nmse = 0., 0.
    running_ssim, running_coil = -1., 0.
    Val_SSIM = []

    result_dir =  target_dir + 'personalized_fastMRIsite%d/pats_%d/totalholdout_%d_pat_%d/LR_%.6f/' %(site,num_train_pats,num_holdout_pats, holdout_num, LR)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for round_idx in range(100):


        if round_idx == 0:
                # !!! Checkpoint to see what performance would be in the very begining before fine tuning
            model.eval()
            val_ssim = []
            val_nmse = []
            with torch.no_grad():
                for sample_idx, sample in tqdm(enumerate(val_loader)):
                    # Move to CUDA
                    for key in sample.keys():
                        try:
                            sample[key] = sample[key].cuda()
                        except:
                            pass
                    # Get outputs
                    est_img_kernel, est_map_kernel, est_ksp = \
                        model(sample, hparams.meta_unrolls, 1)
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
                    est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                        sample['gt_ref_rss'].shape[-1])
                    gt_rss       = sample['gt_ref_rss']
                    data_range   = sample['data_range']

                    # SSIM loss with crop
                    val_ssim.append(ssim(est_crop_rss[:, None],
                                         gt_rss[:, None], data_range).item())
                    val_nmse.append(nmse_loss(gt_rss[:, None],
                                         est_crop_rss[:, None]).item())

            # Final save
            torch.save({
                        'val_ssim': val_ssim,
                        'val_nmse': val_nmse,
                        'model_state_dict': model.state_dict()},
                       result_dir + 'personalized_initial.pt' )

            model.eval()
            hold_ssim = []
            hold_nmse = []
            with torch.no_grad():
                for sample_idx, sample in tqdm(enumerate(holdout_loader)):
                    # Move to CUDA
                    for key in sample.keys():
                        try:
                            sample[key] = sample[key].cuda()
                        except:
                            pass
                    # Get outputs
                    est_img_kernel, est_map_kernel, est_ksp = \
                        model(sample, hparams.meta_unrolls, 1)
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
                    est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                        sample['gt_ref_rss'].shape[-1])
                    gt_rss       = sample['gt_ref_rss']
                    data_range   = sample['data_range']

                    # SSIM loss with crop
                    hold_ssim.append(ssim(est_crop_rss[:, None],
                                         gt_rss[:, None], data_range).item())
                    hold_nmse.append(nmse_loss(gt_rss[:, None],
                                         est_crop_rss[:, None]).item())

            # Final save
            torch.save({
                        'hold_ssim': hold_ssim,
                        'hold_nmse': hold_nmse,
                        'holdout_pats': holdout_pat_files,
                        'train_pats': all_train_files},
                       result_dir + 'holdout_personalized_initial.pt' )




        model.train()
        # Train for one epoch
        for idx, sample in tqdm(enumerate(train_loader)):
            # Move to CUDA
            for key in sample.keys():
                try:
                    sample[key] = sample[key].cuda()
                except:
                    pass

            # Get outputs
            est_img_kernel, est_map_kernel, est_ksp = \
                model(sample, hparams.meta_unrolls, 1)
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
            est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                sample['gt_ref_rss'].shape[-1])
            gt_rss       = sample['gt_ref_rss']
            data_range   = sample['data_range']


            # SSIM loss with crop
            ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)
            loss = hparams.ssim_lam * ssim_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()


            # Other losses for tracking
            with torch.no_grad():
                coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])
                pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                nmse      = nmse_loss(gt_rss,est_crop_rss)

            # Keep a running loss
            running_training = 0.99 * running_training + \
                0.01 * loss.item() if running_training > 0. else loss.item()
            running_ssim = 0.99 * running_ssim + \
                0.01 * (1-ssim_loss.item()) if running_ssim > -1. else (1-ssim_loss.item())
            running_loss = 0.99 * running_loss + \
                0.01 * pix_loss.item() if running_loss > 0. else pix_loss.item()
            running_coil = 0.99 * running_coil + \
                0.01 * coil_loss.item() if running_coil > 0. else coil_loss.item()
            running_nmse = 0.99 * running_nmse + \
                0.01 * nmse.item() if running_nmse > 0. else nmse.item()

            # Logs
            training_log.append(running_training)
            loss_log.append(running_loss)
            ssim_log.append(running_ssim)
            coil_log.append(running_coil)
            nmse_log.append(running_nmse)



            # For MoDL, clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
            optimizer.step()




            # Verbose
            print('Round %d, site %d, Step %d, Batch loss %.4f. Avg. SSIM %.4f, \
    Avg. RSS %.4f, Avg. Coils %.4f, Avg. NMSE %.4f' % (
                round_idx, site, idx, loss.item(),
                running_ssim, running_loss, running_coil,
                running_nmse))






        #     # !!! Checkpoint every N steps
        # model.eval()
        # val_ssim = []
        # val_nmse = []
        # with torch.no_grad():
        #     for sample_idx, sample in tqdm(enumerate(val_loader)):
        #         # Move to CUDA
        #         for key in sample.keys():
        #             try:
        #                 sample[key] = sample[key].cuda()
        #             except:
        #                 pass
        #         # Get outputs
        #         est_img_kernel, est_map_kernel, est_ksp = \
        #             model(sample, hparams.meta_unrolls, 1)
        #         # Extra padding with zero lines - to restore resolution
        #         est_ksp_padded = F.pad(est_ksp, (
        #                 torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
        #                 torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))
        #         # Convert to image domain
        #         est_img_coils = ifft(est_ksp_padded)
        #
        #         # RSS images
        #         est_img_rss = torch.sqrt(
        #             torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))
        #
        #         # Central crop
        #         est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
        #                             sample['gt_ref_rss'].shape[-1])
        #         gt_rss       = sample['gt_ref_rss']
        #         data_range   = sample['data_range']
        #
        #         # SSIM loss with crop
        #         val_ssim.append(ssim(est_crop_rss[:, None],
        #                              gt_rss[:, None], data_range).item())
        #         val_nmse.append(nmse_loss(gt_rss[:, None],
        #                              est_crop_rss[:, None]).item())
        #
        # # Final save
        # torch.save({'ssim_log': ssim_log,
        #             'loss_log': loss_log,
        #             'coil_log': coil_log,
        #             'nmse_log': nmse_log,
        #             'loss': loss,
        #             'val_ssim': val_ssim,
        #             'val_nmse': val_nmse,
        #             'model_state_dict': model.state_dict()},
        #            result_dir + 'personalized_round%d.pt' %(round_idx))



            # !!! Checkpoint every N steps
        model.eval()
        hold_ssim = []
        hold_nmse = []
        with torch.no_grad():
            for sample_idx, sample in tqdm(enumerate(holdout_loader)):
                # Move to CUDA
                for key in sample.keys():
                    try:
                        sample[key] = sample[key].cuda()
                    except:
                        pass
                # Get outputs
                est_img_kernel, est_map_kernel, est_ksp = \
                    model(sample, hparams.meta_unrolls, 1)
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
                est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                    sample['gt_ref_rss'].shape[-1])
                gt_rss       = sample['gt_ref_rss']
                data_range   = sample['data_range']

                # SSIM loss with crop
                hold_ssim.append(ssim(est_crop_rss[:, None],
                                     gt_rss[:, None], data_range).item())
                hold_nmse.append(nmse_loss(gt_rss[:, None],
                                     est_crop_rss[:, None]).item())

        # Final save
        torch.save({
                    'hold_ssim': hold_ssim,
                    'hold_nmse': hold_nmse,
                    'model_state_dict': model.state_dict()},
                   result_dir + 'holdout_personalized_round%d.pt' %(round_idx))
