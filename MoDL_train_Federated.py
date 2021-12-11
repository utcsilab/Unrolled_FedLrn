#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

import torch, os, glob, copy
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

from datagen import MCFullFastMRI, crop
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
from site_loader import site_loader

def get_args():
    parser = OptionParser()
    parser.add_option('--pats'    , '--pats'    , nargs = '+',type='int', default=[5,10,20,50,100]   , help = '# of patients')
    parser.add_option('--sites'    , '--sites'    , nargs = '+',type='int', default=[0,5]    , help='site #s you wish to train') #this script only supports one site
    parser.add_option('--seed'    , '--seed'    , type='int', default=1500, help='random seed to use')
    parser.add_option('--GPU'     , '--GPU'     , type='int', default=0   , help='GPU to Use')
    parser.add_option('--num_work', '--num_work', type='int', default=10  , help='number of workers to use')
    parser.add_option('--start_ep', '--start_ep', type='int', default=0   , help='start epoch for training')
    parser.add_option('--end_ep'  , '--end_ep'  , type='int', default=40 , help='end epoch for training')
    parser.add_option('--ch'  , '--ch'  , type='int', default=18 , help='number of channels in Unet')
    parser.add_option('--num_pool'  , '--num_pool'  , type='int', default=4 , help='number of pool layer in UNet')
    parser.add_option('--LR'  , '--LR'  , type='float', default=3e-4 , help='learning rate for training')
    parser.add_option('--comp_val'  , '--comp_val'  , type='int', default=1 , help='do you want to compute validation results every epoch')
    parser.add_option('--share_int'  , '--share_int'  , type='int', default=1 , help='how often do we share weights')

    (options, args) = parser.parse_args()
    return options


args = get_args()
print(args)

GPU_ID                = args.GPU
global_seed           = args.seed
num_train_pats        = args.pats
num_workers           = args.num_work
sites                 = args.sites
start_epoch           = args.start_ep
end_epoch             = args.end_ep
lr                    = args.LR
unet_ch               = args.ch
unet_num_pool         = args.num_pool
comp_val              = args.comp_val
sharing_interval      = args.share_int
num_val_pats = 10
plt.rcParams.update({'font.size': 12})
plt.ioff(); plt.close('all')

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)


n_stdev =  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 'num_slices' around 'central_slice' from each scan. Again this may need to be unique for each dataset
center_slice_knee = 17
num_slices_knee   = 10
center_slice_brain = 5
num_slices_brain = 10

for train_pats in num_train_pats:
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
    ############################################################################
    # Mode-specific settings
    if hparams.mode == 'MoDL':
        hparams.use_map_net     = False # No Map-net
        hparams.map_init        = 'espirit'
        hparams.block1_max_iter = 0 # Maps

    # Not used just a hold over from previous code
    if hparams.mode == 'DeepJSense':
        hparams.use_map_net     = True
        hparams.map_init        = 'estimated' # Can be anything
        hparams.block1_max_iter = 6 # Maps
        train_maps, val_maps    = None, None
        # Map network parameters
        hparams.map_channels = 64
        hparams.map_blocks   = 4
    ##################### Image- and Map-Net parameters#########################
    hparams.img_arch     = 'UNet' # 'UNet' or 'ResNet' or 'ResNetSplit' for same split in all unrolls or 'ResUnrollSplit' to split across unrolls
    hparams.img_channels = unet_ch
    hparams.img_blocks   = unet_num_pool
    if hparams.img_arch != 'UNet':
        hparams.latent_channels = 64
    hparams.downsample   = 4 # Training R
    hparams.kernel_size  = 3
    hparams.in_channels     = 2
    ######################Specify Weight Sharing Method#########################
    hparams.share_mode = 'Full_avg' # either Seperate, Full_avg or Split_avg for how we want to share weights
    ##########################################################################
    if hparams.share_mode != 'Seperate':
    #the following parameters are only really used if using ReNetSplit for FedLrning. They specify how you want to split the ResNet
        hparams.img_sep         = False # Do we use separate networks at each unroll?
        hparams.all_sep         = False  #allow all unrolls to be unique within model(True) or create just 2 networks(False): Local & Global
        hparams.num_blocks1     = 4
        hparams.num_blocks2     = 2
        hparams.share_global    = 2
    ###########################################################################
    # MoDL parametrs
    #NOTE: some of the params are used only for DeepJSense
    hparams.use_img_net        = True
    hparams.img_init           = 'estimated'
    hparams.mps_kernel_shape   = [15, 15, 9] # Always 15 coils
    hparams.l2lam_init         = 0.1
    hparams.l2lam_train        = True
    hparams.meta_unrolls       = 6 # Ending value - main value
    hparams.block2_max_iter    = 6 # Image
    hparams.cg_eps             = 1e-6
    hparams.verbose            = False
    # Static training parameters
    hparams.lr           = lr # Finetune if desired
    hparams.step_size    = 20    # Number of epochs to decay with gamma
    hparams.decay_gamma  = 0.5
    hparams.grad_clip    = 1. # Clip gradients
    hparams.start_epoch  = 0 # Warm start from a specific epoch
    hparams.batch_size   = 1 # !!! Unsupported !!!
    # Loss lambdas
    hparams.coil_lam = 0.
    hparams.ssim_lam = 1.
    #################Set save location directory################################
    global_dir = 'Results/multi_site_FedLrn/%d_sites/sharing_interval_%d/%s/%s/num_pool%d_num_ch%d/num_train_patients%d/seed%d' % (
          len(sites),sharing_interval,hparams.mode, hparams.img_arch,hparams.img_blocks, hparams.img_channels,
         train_pats, global_seed )

    if not os.path.exists(global_dir):
        os.makedirs(global_dir)
    ######################initialize model using hparams########################
    model = MoDLDoubleUnroll(hparams)
    model  = model.cuda()
    # Switch to train
    model.train()
    torch.save({
        'model': model.state_dict(),
        }, global_dir + '/Initial_weights.pt')
    # Count parameters
    total_params = np.sum([np.prod(p.shape) for p
                           in model.parameters() if p.requires_grad])
    print('Total parameters %d' % total_params)
    ############################################################################
    ########Get relevant data and datloaders Local training datasets############
    train_loaders = []
    val_loaders   = []
    for i in range(len(sites)):
        if sites[i] <=3: #must be a knee site
            center_slice = center_slice_knee
            num_slices = num_slices_knee
        elif sites[i] >3: #must be a brain site
            center_slice = center_slice_brain
            num_slices = num_slices_brain

        #get list of samples from function
        train_files, train_maps, val_files, val_maps = site_loader(sites[i], train_pats, num_val_pats)
        print(train_files[0])
        train_dataset = MCFullFastMRI(train_files, num_slices, center_slice,
                                        downsample=hparams.downsample,
                                        mps_kernel_shape=hparams.mps_kernel_shape,
                                        maps=train_maps, mask_params=train_mask_params, noise_stdev = 0.0)
        train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                         shuffle=True, num_workers=num_workers, drop_last=True)
        train_loaders.append(train_loader)
        # Local Validation datasets
        val_dataset = MCFullFastMRI(val_files, num_slices, center_slice,
                                        downsample=hparams.downsample,
                                        mps_kernel_shape=hparams.mps_kernel_shape,
                                        maps=val_maps, mask_params=val_mask_params, noise_stdev = 0.0)
        val_loader  = DataLoader(val_dataset, batch_size=hparams.batch_size,
                                     shuffle=True, num_workers=num_workers, drop_last=True)
        val_loaders.append(val_loader)
    ############################################################################

    ############################################################################
    #Training relevent information
    # Criterions
    ssim           = SSIMLoss().cuda()
    multicoil_loss = MCLoss().cuda()
    pixel_loss     = torch.nn.MSELoss(reduction='sum')
    nmse_loss      = NMSELoss()

    # Get optimizer and scheduler(One for each site)
    optimizers = []
    schedulers = []
    for i in range(len(sites)):
        optimizer = Adam(model.parameters(), lr=hparams.lr)
        scheduler = StepLR(optimizer, hparams.step_size,
                           gamma=hparams.decay_gamma)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

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
    Val_SSIM = []
    for i in range(len(sites)):
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
        Val_SSIM.append([])

    local_dir = global_dir + '/N%d_n%d_lamInit%.3f' % (
            hparams.meta_unrolls, hparams.block1_max_iter,
            hparams.l2lam_init)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    ############################################################################
    #######end of setting training configurations###############################
    ############################################################################


    #############Training Occurs Below#######
    model.train()
# For each epoch
    for epoch_idx in range(start_epoch, end_epoch):
        # Log one divergence event per epoch
        first_time, stable_model, stable_opt = True, None, None

        for site in range(len(sites)):
            if epoch_idx == 0:
                # if this is the first epoch then every sites model needs to start from the same initialization
                print('loading initial weights')
                saved_model = torch.load(global_dir + '/Initial_weights.pt')
                model.load_state_dict(saved_model['model'])
            else:
                #load relavent model information for the current site to be trained
                saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx-1)+  'site'+str(sites[site])+'after_share_weights.pt')
                model.load_state_dict(saved_model['model_state_dict'])
                if hparams.start_epoch>0 and epoch_idx == hparams.star_epoch:
                    #load optimizer states and schedulers if starting from a later epoch
                    optimizers[site].load_state_dict(saved_model['optimizer_state_dict'])
                    schedulers[site].load_state_dict(saved_model['scheduler_state_dict'])

            for sample_idx, sample in tqdm(enumerate(train_loaders[site])):
                # Move to CUDA
                for key in sample.keys():
                    try:
                        sample[key] = sample[key].cuda()
                    except:
                        pass
                # Get outputs
                est_img_kernel, est_map_kernel, est_ksp = \
                    model(sample, hparams.meta_unrolls,
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
                est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                    sample['gt_ref_rss'].shape[-1])
                gt_rss       = sample['gt_ref_rss']
                data_range   = sample['data_range']

                # SSIM loss with crop
                ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)
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

                # Backprop
                optimizers[site].zero_grad()
                loss.backward()
                # For MoDL (?), clip gradients
                torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
                optimizers[site].step()

               # Verbose
                print('Epoch %d, Site %d ,Step %d, Batch loss %.4f. Avg. SSIM %.4f, Avg. RSS %.4f, Avg. Coils %.4f, Avg. NMSE %.4f' % (epoch_idx,
                sites[site], sample_idx, loss.item(), running_ssim[site], running_loss[site], running_coil[site], running_nmse[site]))
            torch.save({
                'epoch': epoch_idx,
                'sample_idx': sample_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizers[site].state_dict(),
                'scheduler_state_dict': schedulers[site].state_dict(),
                'ssim_log': ssim_log,
                'loss_log': loss_log,
                'coil_log': coil_log,
                'nmse_log': nmse_log,
                'loss': loss,
                'hparams': hparams,
                'train_mask_params': train_mask_params}, local_dir + '/epoch'+str(epoch_idx)+'_site'+str(sites[site])+'last_weights.pt')

        #take a step with all schedulers
        for i in range(len(sites)):
            schedulers[i].step()


        ##################FEDERATED WEIGHT SHARING##############################
        with torch.no_grad():
            if epoch_idx%sharing_interval == 0:
                print("Federated weight averaging occuring")
                state_dict_accumulator = MoDLDoubleUnroll(hparams)
                #print("device:" , state_dict_accumulator.get_device())
                saved_model = torch.load(local_dir + '/epoch'+str(epoch_idx)+'_site'+str(sites[0])+'last_weights.pt') # load the first site initially

                state_dict_accumulator.load_state_dict(saved_model['model_state_dict'])
                state_dict_accumulator = state_dict_accumulator.image_net.state_dict()

                # state_dict_transitive.load_state_dict(saved_model['model' + str(0) + '_state_dict'])
                # state_dict_transitive = state_dict_transitive.image_net.state_dict()

                for key in state_dict_accumulator:
                    for site in range(len(sites)-1):
                        state_dict_transitive  = MoDLDoubleUnroll(hparams)
                        temp_model = torch.load(local_dir + '/epoch'+str(epoch_idx)+'_site'+str(sites[site+1])+'last_weights.pt')
                        state_dict_transitive.load_state_dict(temp_model['model_state_dict'])
                        state_dict_transitive = state_dict_transitive.image_net.state_dict()

                        state_dict_accumulator[key]  = (state_dict_accumulator[key]  + state_dict_transitive[key])

                    state_dict_accumulator[key] = state_dict_accumulator[key]/len(sites)

                for site in range(len(sites)):
                    model.image_net.load_state_dict(state_dict_accumulator) #save the averaged weights to each sites unique file

                    torch.save({
                        'epoch': epoch_idx,
                        'sample_idx': sample_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizers[site].state_dict(),
                        'scheduler_state_dict': schedulers[site].state_dict(),
                        'ssim_log': ssim_log[site],
                        'loss_log': loss_log[site],
                        'coil_log': coil_log[site],
                        'nmse_log': nmse_log[site],
                        'loss': loss,
                        'hparams': hparams,
                        'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(sites[site])+'after_share_weights.pt')

            else:
                print("No sharing this round")
                for site in range(len(sites)):
                    saved_model = torch.load(local_dir + '/epoch'+str(epoch_idx)+'_site'+str(sites[site])+'last_weights.pt')
                    model.load_state_dict(saved_model['model_state_dict'])
                    torch.save({
                        'epoch': epoch_idx,
                        'sample_idx': sample_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizers[site].state_dict(),
                        'scheduler_state_dict': schedulers[site].state_dict(),
                        'ssim_log': ssim_log[site],
                        'loss_log': loss_log[site],
                        'coil_log': coil_log[site],
                        'nmse_log': nmse_log[site],
                        'loss' : loss,
                        'hparams': hparams,
                        'train_mask_params': train_mask_params}, local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(sites[site])+'after_share_weights.pt')




        ########################################################################
        for site in range(len(sites)):
            #compute validation SSIM
            # After each epoch, check some validation samples
            saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(sites[site])+'after_share_weights.pt')
            model.load_state_dict(saved_model['model_state_dict'])
            if comp_val:
                model.eval()
                running_SSIM_val = 0.0
                for sample_idx, sample in tqdm(enumerate(val_loaders[site])):
                    # Move to CUDA
                    for key in sample.keys():
                        try:
                            sample[key] = sample[key].cuda()
                        except:
                            pass
                    with torch.no_grad():
                        # Get outputs
                        est_img_kernel, est_map_kernel, est_ksp = \
                            model(sample, hparams.meta_unrolls,
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
                        est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                            sample['gt_ref_rss'].shape[-1])
                        gt_rss       = sample['gt_ref_rss']
                        data_range   = sample['data_range']

                        # SSIM loss with crop
                        ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)
                        # coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])
                        # loss = hparams.ssim_lam * ssim_loss + hparams.coil_lam * coil_loss
                        # # Other Loss for tracking
                        # pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                        # nmse      = nmse_loss(gt_rss,est_crop_rss)
                        running_SSIM_val = running_SSIM_val + (1-ssim_loss.item())
                if sites[0] <=3:
                    Val_SSIM[site].append(running_SSIM_val/(num_slices_knee*num_val_pats))
                elif sites[0]>3:
                    Val_SSIM[site].append(running_SSIM_val/(num_slices_brain*num_val_pats))
                plt.figure()
                # Plot
                plt.subplot(1,2,1)
                plt.title('Training SSIM')
                plt.plot(np.asarray(ssim_log[site]))
                plt.subplot(1,2,2)
                plt.title('Validation SSIM')
                plt.plot(np.asarray(Val_SSIM[site]))

                # Save
                plt.tight_layout()
                plt.savefig(local_dir + '/train&val_curves_site' + str(sites[site]) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()

                #compute sample validation images

                #not sure why but I keep getting different smaples every epoch if I dont set the seed here every epoch
                torch.manual_seed(global_seed)
                np.random.seed(global_seed)

                # After each epoch, check some validation samples
                saved_model = torch.load(local_dir + '/ckpt_epoch' +str(epoch_idx)+  'site'+str(sites[site])+'after_share_weights.pt')
                model.load_state_dict(saved_model['model_state_dict'])

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
                            model(sample, hparams.meta_unrolls)

                        # Extra padding with dead zones
                        est_ksp_padded = F.pad(est_ksp, (
                            torch.sum(sample['dead_lines'] < est_ksp.shape[-1]//2).item(),
                            torch.sum(sample['dead_lines'] > est_ksp.shape[-1]//2).item()))

                        # Convert to image domain
                        est_img_coils = ifft(est_ksp_padded)

                        # RSS images
                        est_img_rss = torch.sqrt(torch.sum(torch.square(torch.abs(est_img_coils)), axis=1))
                        # Central crop
                        est_crop_rss = crop(est_img_rss, sample['gt_ref_rss'].shape[-2],
                                            sample['gt_ref_rss'].shape[-1])
                        # Losses
                        ssim_loss = ssim(est_crop_rss[:, None], sample['gt_ref_rss'][:, None],
                                         sample['data_range'])
                        l1_loss   = pixel_loss(est_crop_rss, sample['gt_ref_rss'])

                    # Plot
                    plt.subplot(2, 4, sample_idx+1)
                    plt.imshow(torch.flip(sample['gt_ref_rss'][0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(sample['gt_ref_rss'][0]), cmap='gray')
                    plt.axis('off'); plt.title('GT RSS')
                    plt.subplot(2, 4, sample_idx+1+4*1)
                    plt.imshow(torch.flip(est_crop_rss[0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(est_crop_rss[0]), cmap='gray')
                    plt.axis('off'); plt.title('Ours - RSS')

                # Save
                plt.tight_layout()
                plt.savefig(local_dir + '/val_site' + str(sites[site]) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()



        model.train()
