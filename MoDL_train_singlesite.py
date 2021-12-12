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

from argparse import ArgumentParser
from site_loader import site_loader

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--pats'    , nargs = '+', type=int, default=[5,100], help = '# of patients')
    parser.add_argument('--sites'   , nargs = '+', type=int, default=[3]    , help='site #s you wish to train') #this script only supports one site
    parser.add_argument('--seed'    , type=int, default=1500                , help='random seed to use')
    parser.add_argument('--GPU'     , type=int, default=1                   , help='GPU to Use')
    parser.add_argument('--num_work', type=int, default=1                   , help='number of workers to use')
    parser.add_argument('--start_ep', type=int, default=0                   , help='start epoch for training')
    parser.add_argument('--end_ep'  , nargs='+', type=int, default=[800, 40], help='end epoch for training')
    parser.add_argument('--ch'      , type=int, default=18                  , help='number of channels in Unet')
    parser.add_argument('--num_pool', type=int, default=4                   , help='number of pool layer in UNet')
    parser.add_argument('--LR'      , type=float, default=3e-4              , help='learning rate for training')
    parser.add_argument('--decay_ep', nargs='+', type=int, default=[400, 20], help='number of epochs after which LR is decayed')
    parser.add_argument('--save_interval', nargs='+', type=int, default=[20, 1], help='save once every this many epochs')
    parser.add_argument('--comp_val', type=int, default=1                   , help='do you want to compute validation results every epoch')

    args = parser.parse_args()
    return args

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
decay_epochs          = args.decay_ep
save_interval         = args.save_interval
unet_ch               = args.ch
unet_num_pool         = args.num_pool

num_val_pats = 28
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

for train_pats, val_interval, num_epochs, decay_ep in \
    zip(num_train_pats, save_interval, end_epoch, decay_epochs):
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
    hparams.in_channels  = 2
    
    ######################Specify Weight Sharing Method#########################
    hparams.share_mode = 'Seperate' # either Seperate, Full_avg or Split_avg for how we want to share weights
    ##########################################################################
    if hparams.share_mode != 'Seperate':
    #the following parameters are only really used if using ReNetSplit for FedLrning. They specify how you want to split the ResNet
        hparams.img_sep         = False # Do we use separate networks at each unroll?
        hparams.all_sep         = False  #allow all unrolls to be unique within model(True) or create just 2 networks(False): Local & Global
        hparams.num_blocks1     = 4
        hparams.num_blocks2     = 2
        hparams.share_global    = 2
    else:
        # TODO: (Marius) Careful here, the stuff wasn't set before 
        # It just happened that img_sep is not 'True', but it wasn't explicitly 'False' either
        hparams.img_sep = False
        
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
    hparams.decay_epochs = decay_ep # Number of epochs to decay with gamma
    hparams.decay_gamma  = 0.5
    hparams.grad_clip    = 1. # Clip gradients
    hparams.start_epoch  = 0 # Warm start from a specific epoch
    hparams.batch_size   = 1 # !!! Unsupported !!!
    
    # Loss lambdas
    hparams.coil_lam = 0.
    hparams.ssim_lam = 1.
    
    #################Set save location directory################################
    global_dir = 'Results/single_site/site_%d/%s/%s/num_pool%d_num_ch%d/num_train_patients%d/seed%d' % (
         sites[0], hparams.mode, hparams.img_arch,hparams.img_blocks, hparams.img_channels,
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
    if sites[0] <=3: #must be a knee site
        center_slice = center_slice_knee
        num_slices = num_slices_knee
    elif sites[0] >3: #must be a brain site
        center_slice = center_slice_brain
        num_slices = num_slices_brain

    #get list of samples from function
    train_files, train_maps, val_files, val_maps = site_loader(sites[0], train_pats, num_val_pats)
    train_dataset = MCFullFastMRI(train_files, num_slices, center_slice,
                                downsample=hparams.downsample,
                                mps_kernel_shape=hparams.mps_kernel_shape,
                                maps=train_maps, mask_params=train_mask_params, noise_stdev = 0.0)
    train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                     shuffle=True, num_workers=num_workers, drop_last=True)

    # Local Validation datasets
    val_dataset = MCFullFastMRI(val_files, num_slices, center_slice,
                                downsample=hparams.downsample,
                                mps_kernel_shape=hparams.mps_kernel_shape,
                                maps=val_maps, mask_params=val_mask_params, noise_stdev = 0.0,
                                scramble=True)
    val_loader  = DataLoader(val_dataset, batch_size=hparams.batch_size,
                             shuffle=False, num_workers=num_workers, drop_last=True)

    ############################################################################
    # Training relevent information
    # Criterions
    ssim           = SSIMLoss().cuda()
    multicoil_loss = MCLoss().cuda()
    pixel_loss     = torch.nn.MSELoss(reduction='sum')
    nmse_loss      = NMSELoss()

    # Get optimizer and scheduler(One for each site)
    optimizer = Adam(model.parameters(), lr=hparams.lr)
    scheduler = StepLR(optimizer, hparams.decay_epochs,
                       gamma=hparams.decay_gamma)

    #create logs for each site in list format
    best_loss = np.inf
    training_log = []
    loss_log = []
    ssim_log = []
    coil_log = []
    nmse_log = []
    running_training = 0.
    running_loss = 0.
    running_nmse = 0.
    running_ssim = -1.
    running_coil = 0.
    Val_SSIM = []

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
    for epoch_idx in range(num_epochs):
        for sample_idx, sample in tqdm(enumerate(train_loader)):
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
            loss      = ssim_loss

            # Other Loss for tracking
            with torch.no_grad():
                coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])
                pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                nmse      = nmse_loss(gt_rss,est_crop_rss)

            # Keep a running loss
            running_training = 0.99 * running_training + 0.01 * loss.item() if running_training > 0. else loss.item()
            running_ssim = 0.99 * running_ssim + 0.01 * (1-ssim_loss.item()) if running_ssim > -1. else (1-ssim_loss.item())
            running_loss = 0.99 * running_loss + 0.01 * pix_loss.item() if running_loss > 0. else pix_loss.item()
            running_coil = 0.99 * running_coil + 0.01 * coil_loss.item() if running_coil > 0. else coil_loss.item()
            running_nmse = 0.99 * running_nmse + 0.01 * nmse.item() if running_nmse > 0. else nmse.item()

            training_log.append(running_training)
            loss_log.append(running_loss)
            ssim_log.append(running_ssim)
            coil_log.append(running_coil)
            nmse_log.append(running_nmse)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
            optimizer.step()

            # Verbose
            print('Epoch %d, Site %d, Step %d, Batch loss %.4f. Avg. SSIM %.4f, Avg. RSS %.4f, Avg. Coils %.4f, Avg. NMSE %.4f' % (
                epoch_idx, sites[0], sample_idx, loss.item(), running_ssim, 
                running_loss, running_coil, running_nmse))

            # Plot the first training sample
            if sample_idx == 0:
                # Plot
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(torch.flip(sample['gt_ref_rss'][0],[0,1]).cpu().detach().numpy(), 
                           vmin=0., vmax=torch.max(sample['gt_ref_rss'][0]), cmap='gray')
                plt.axis('off'); plt.title('GT RSS')
                plt.subplot(1, 2, 2)
                plt.imshow(torch.flip(est_crop_rss[0],[0,1]).cpu().detach().numpy(), 
                           vmin=0., vmax=torch.max(est_crop_rss[0]), cmap='gray')
                plt.axis('off'); plt.title('MoDL - SSIM: %.4f' % (1 - ssim_loss.item()))

                # Save
                plt.tight_layout()
                plt.savefig(local_dir + '/train_site' + str(sites[0]) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()
        
        # Save periodically
        if np.mod(epoch_idx + 1, val_interval) == 0:
            torch.save({
                'epoch': epoch_idx,
                'sample_idx': sample_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ssim_log': ssim_log,
                'loss_log': loss_log,
                'coil_log': coil_log,
                'nmse_log': nmse_log,
                'loss': loss,
                'hparams': hparams,
                'train_mask_params': train_mask_params}, 
                local_dir + '/ckpt_epoch' + str(epoch_idx) + '_site' +
                str(sites[0]) + 'last_weights.pt')

        scheduler.step()

        # Compute validation SSIM
        if args.comp_val and np.mod(epoch_idx + 1, val_interval) == 0:
            # Switch to eval
            model.eval()
            running_SSIM_val = 0.0
            plt.figure()
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
                    
                    # Add the first few ones to plot
                    if sample_idx < 4:
                        plt.subplot(2, 4, sample_idx+1)
                        plt.imshow(torch.flip(sample['gt_ref_rss'][0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(sample['gt_ref_rss'][0]), cmap='gray')
                        plt.axis('off'); plt.title('GT RSS')
                        plt.subplot(2, 4, sample_idx+1+4*1)
                        plt.imshow(torch.flip(est_crop_rss[0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(est_crop_rss[0]), cmap='gray')
                        plt.axis('off'); plt.title('SSIM: %.3f' % (1-ssim_loss.item()))

                    # SSIM loss with crop
                    ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)
                    running_SSIM_val = running_SSIM_val + (1-ssim_loss.item())
                
                # Save
                plt.tight_layout()
                plt.savefig(local_dir + '/val_site' + str(sites[0]) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
                plt.close()
                
                if sites[0] <= 3:
                    Val_SSIM.append(running_SSIM_val/(num_slices_knee*num_val_pats))
                elif sites[0] > 3:
                    Val_SSIM.append(running_SSIM_val/(num_slices_brain*num_val_pats))
                
            # Plot validation metrics
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('Training SSIM')
            plt.plot(np.asarray(ssim_log))
            plt.subplot(1, 2, 2)
            plt.title('Validation SSIM')
            plt.plot(np.asarray(Val_SSIM))
            # Save
            plt.tight_layout()
            plt.savefig(local_dir + '/train&val_curves_site' + str(sites[0]) + '_samples_epoch%d.png' % epoch_idx, dpi=300)
            plt.close()
        # Back to training
        model.train()