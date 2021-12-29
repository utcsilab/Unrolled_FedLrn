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

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

from argparse import ArgumentParser
from site_loader import site_loader

plt.rcParams.update({'font.size': 12})
plt.ioff(); plt.close('all')

# Maybe
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# Always !!!
torch.backends.cudnn.benchmark = True

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--seed'    , type=int, default=1500     , help='random seed to use')
    parser.add_argument('--mini_seed', type=int                  , help='an extra seed to shuffle clients')
    parser.add_argument('--GPU'     , type=int                   , help='GPU to Use')
    parser.add_argument('--num_work', type=int                   , help='number of workers to use')
    parser.add_argument('--train_dilation', type=int             , help='boost num rounds, save interval and decay in 24:1:16 ratio')
    parser.add_argument('--ch'      , type=int                   , help='number of channels in Unet')
    parser.add_argument('--num_pool', type=int                   , help='number of pool layer in UNet')
    parser.add_argument('--LR'      , type=float, default=3e-4   , help='learning rate for training')
    parser.add_argument('--client_pats' , nargs='+', type=int, help='Vector of client samples (patients, 10 slices each)')
    parser.add_argument('--client_sites', nargs='+', type=int, help='Vector of client sites (local data distribution)')
    parser.add_argument('--share_int'   , '--share_int', type= int, default=12, help='how often do we share weights(measured in steps)')

    args = parser.parse_args()
    return args

# Get arguments
args = get_args()
print(args)

GPU_ID                = args.GPU
global_seed           = args.seed
mini_seed             = args.mini_seed
num_workers           = args.num_work
lr                    = args.LR
unet_ch               = args.ch
unet_num_pool         = args.num_pool

# Inferred in the ratio 24:1:16
num_rounds            = args.train_dilation * 24
save_interval         = args.train_dilation * 1
decay_epochs          = args.train_dilation * 16

share_int             = args.share_int
client_pats           = args.client_pats
client_sites          = args.client_sites
num_val_pats          = 20
# Immediately determine number of clients
assert len(client_pats) == len(client_sites), 'Client args mismatch!'
num_clients = len(client_pats)
np.random.seed(mini_seed)
client_perm = np.random.permutation(num_clients)

# Determine if this is a multi-site setup or not
# Currently just used to name a folder
if len(np.unique(client_sites)) == 1:
    multi_site = False
else:
    multi_site = True

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# 'num_slices' around 'central_slice' from each scan. Again this may need to be unique for each dataset
center_slice_knee  = 17
num_slices_knee    = 10
center_slice_brain = 5
num_slices_brain   = 10

# Set seed
torch.manual_seed(global_seed)
np.random.seed(global_seed)

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
    assert False, 'Deprecated!'

##################### Image- and Map-Net parameters#########################
hparams.img_arch     = 'UNet' # 'UNet' only
hparams.img_channels = unet_ch
hparams.img_blocks   = unet_num_pool
if hparams.img_arch != 'UNet':
    hparams.latent_channels = 64
hparams.downsample   = 4 # Training R
hparams.kernel_size  = 3
hparams.in_channels  = 2

######################Specify Architecture Sharing Method#########################
hparams.share_mode = 'Seperate' # either Seperate, Full_avg or Split_avg for how we want to share weights
##########################################################################
if hparams.share_mode != 'Seperate':
#cthe following parameters are only really used if using ReNetSplit for FedLrning. They specify how you want to split the ResNet
    hparams.img_sep         = False # Do we use separate networks at each unroll?
    hparams.all_sep         = False # allow all unrolls to be unique within model(True) or create just 2 networks(False): Local & Global
    hparams.num_blocks1     = 4
    hparams.num_blocks2     = 2
    hparams.share_global    = 2
else:
    hparams.img_sep = False

###########################################################################
# MoDL parametrs
# NOTE: some of the params are used only for DeepJSense
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
hparams.decay_epochs = decay_epochs # Number of epochs to decay with gamma
hparams.decay_gamma  = 0.5
hparams.grad_clip    = 1. # Clip gradients
hparams.start_epoch  = 0 # Warm start from a specific epoch
hparams.batch_size   = 1 # !!! Unsupported !!!

# Loss lambdas
hparams.coil_lam = 0.
hparams.ssim_lam = 1.
# print(multi_site, num_clients, share_int, hparams.img_arch, hparams.img_blocks, hparams.img_channels, np.min(client_pats), global_seed)
#################Set save location directory################################
global_dir = 'Results/federated/multiSite%d_clients%d_siteMin%d_siteMax%d/\
sync%d/%s_pool%d_ch%d/num_train_patients%d/seed%d' % (
      multi_site, num_clients, np.min(client_sites),
      np.max(client_sites), share_int, hparams.img_arch,
      hparams.img_blocks, hparams.img_channels,
      np.min(client_pats), global_seed)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)

######################initialize model using hparams########################
model = MoDLDoubleUnroll(hparams)
model = model.cuda()
# Initialize scratch model
scratch_model = MoDLDoubleUnroll(hparams)

# Switch to train
model.train()
# Count parameters
total_params = np.sum([np.prod(p.shape) for p
                       in model.parameters() if p.requires_grad])
print('Total parameters %d' % total_params)

######## Get client datasets and dataloaders ############
train_loaders, val_loaders  = [None for idx in range(num_clients)], \
    [None for idx in range(num_clients)]
iterators = [None for idx in range(num_clients)]
cursors   = [] #make a cursor for each unique site(used (not client)
unique_sites = list(set(client_sites))
num_unique = len(unique_sites)
print('unique sites:', unique_sites)
print('# of unique sites:', num_unique)
for i in range(num_unique):
    cursors.append(0)

# Get unique data for each client, from their own site
for i in client_perm:
    # Get all the filenames from the site
    all_train_files, all_train_maps, all_val_files, all_val_maps = \
        site_loader(client_sites[i], 100, num_val_pats)#always load the maximum number of patients for that site then parse it down

    unique_site_idx = unique_sites.index(client_sites[i]) #get site index for cursor
    print('unique site idx:', unique_site_idx)
    # Subselect unique (by incrementing) training samples
    client_idx = np.arange(cursors[unique_site_idx], 
           cursors[unique_site_idx] + client_pats[i]).astype(int)
    print('client_idx:', client_idx)
    print('client files:', [all_train_files[j] for j in client_idx])

    client_train_files, client_train_maps = \
        [all_train_files[j] for j in client_idx], [all_train_maps[j] for j in client_idx]

    # Increment cursor for the site
    cursors[unique_site_idx] = cursors[unique_site_idx] + client_pats[i]

    # Maintenance
    if client_sites[i] <= 3: # must be a knee site
        center_slice = center_slice_knee
        num_slices   = num_slices_knee
    elif client_sites[i] > 3: # must be a brain site
        center_slice = center_slice_brain
        num_slices   = num_slices_brain

    # Create client dataset and loader
    train_dataset = MCFullFastMRI(client_train_files, num_slices, center_slice,
                                  downsample=hparams.downsample,
                                  mps_kernel_shape=hparams.mps_kernel_shape,
                                  maps=client_train_maps, mask_params=train_mask_params,
                                  noise_stdev = 0.0)
    train_loader  = DataLoader(train_dataset, batch_size=hparams.batch_size,
                               shuffle=True, num_workers=num_workers, drop_last=True,
                               pin_memory=True)
    train_loaders[i] = copy.deepcopy(train_loader)
    iterators[i]     = iter(train_loaders[i])

    # Create client dataset and loader
    val_dataset = MCFullFastMRI(all_val_files, num_slices, center_slice,
                                downsample=hparams.downsample,
                                mps_kernel_shape=hparams.mps_kernel_shape,
                                maps=all_val_maps, mask_params=val_mask_params,
                                noise_stdev = 0.0, scramble=True)
    val_loader  = DataLoader(val_dataset, batch_size=hparams.batch_size,
                             shuffle=False, num_workers=num_workers, drop_last=True)
    val_loaders[i] = copy.deepcopy(val_loader)

# Criterions
ssim           = SSIMLoss().cuda()
multicoil_loss = MCLoss().cuda()
pixel_loss     = torch.nn.MSELoss(reduction='sum')
nmse_loss      = NMSELoss()

# Get optimizer and scheduler (One for each site)
optimizers = []
schedulers = []
for i in range(num_clients):
    optimizer = Adam(model.parameters(), lr=hparams.lr)
    scheduler = StepLR(optimizer, hparams.decay_epochs,
                       gamma=hparams.decay_gamma)
    optimizers.append(optimizer)
    schedulers.append(scheduler)

# create logs for each client in list format
best_loss, training_log = [], []
loss_log, ssim_log      = [], []
coil_log, nmse_log      = [], []
running_training        = []
running_loss, running_nmse = [], []
running_ssim, running_coil = [], []
Val_SSIM = [] # !!! Only supports one 'client' = downloaded model

for i in range(num_clients):
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

# Another directory
local_dir = global_dir + '/N%d_n%d_lamInit%.3f' % (
        hparams.meta_unrolls, hparams.block1_max_iter,
        hparams.l2lam_init)
if not os.path.isdir(local_dir):
    os.makedirs(local_dir)
# Save initial weights
torch.save({
    'model': model.state_dict(),
    }, local_dir + '/Initial_weights.pt')

# !!! Federation happens via these files
download_file = local_dir + '/fed_download.pt'
upload_files  = [local_dir + '/fed_upload%d.pt' % idx for idx in range(num_clients)]

# For each training-communication round
for round_idx in range(num_rounds):

    # Train the model for the given number of steps at each client
    for client_idx in range(num_clients):
        if round_idx == 0:
            # if this is the first round then every client model starts from the same initialization
            print('loading initial weights')
            saved_model = torch.load(local_dir + '/Initial_weights.pt')
            model.load_state_dict(saved_model['model'])
        else:
            # 'Download 'weights from server
            saved_model = torch.load(download_file)
            model.load_state_dict(saved_model['model_state_dict'])
            # !!! Warm start deprecated

        # Local updates performed by client for a number of steps
        for sample_idx in tqdm(range(share_int)):
            try:
                # Get next batch
                sample = next(iterators[client_idx])
            except:
                # Reset datasets
                print('reset iterator for client %d' % client_idx)
                iterators[client_idx] = iter(train_loaders[client_idx])
                sample                = next(iterators[client_idx] )

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
            loss = hparams.ssim_lam * ssim_loss

            # Other losses for tracking
            with torch.no_grad():
                coil_loss = multicoil_loss(est_ksp, sample['gt_nonzero_ksp'])
                pix_loss  = pixel_loss(est_crop_rss, gt_rss)
                nmse      = nmse_loss(gt_rss,est_crop_rss)

            # Keep a running loss
            running_training[client_idx] = 0.99 * running_training[client_idx] + \
                0.01 * loss.item() if running_training[client_idx] > 0. else loss.item()
            running_ssim[client_idx] = 0.99 * running_ssim[client_idx] + \
                0.01 * (1-ssim_loss.item()) if running_ssim[client_idx] > -1. else (1-ssim_loss.item())
            running_loss[client_idx] = 0.99 * running_loss[client_idx] + \
                0.01 * pix_loss.item() if running_loss[client_idx] > 0. else pix_loss.item()
            running_coil[client_idx] = 0.99 * running_coil[client_idx] + \
                0.01 * coil_loss.item() if running_coil[client_idx] > 0. else coil_loss.item()
            running_nmse[client_idx] = 0.99 * running_nmse[client_idx] + \
                0.01 * nmse.item() if running_nmse[client_idx] > 0. else nmse.item()

            # Logs
            training_log[client_idx].append(running_training[client_idx])
            loss_log[client_idx].append(running_loss[client_idx])
            ssim_log[client_idx].append(running_ssim[client_idx])
            coil_log[client_idx].append(running_coil[client_idx])
            nmse_log[client_idx].append(running_nmse[client_idx])

            # Backprop
            optimizers[client_idx].zero_grad()
            loss.backward()
            # For MoDL, clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(), hparams.grad_clip)
            optimizers[client_idx].step()

            # Verbose
            print('Round %d, Client %d, Step %d, Batch loss %.4f. Avg. SSIM %.4f, \
Avg. RSS %.4f, Avg. Coils %.4f, Avg. NMSE %.4f' % (
                round_idx, client_idx, sample_idx, loss.item(),
                running_ssim[client_idx], running_loss[client_idx], running_coil[client_idx],
                running_nmse[client_idx]))

        # After each round, save the local weights of all clients before downloading
        if os.path.exists(upload_files[client_idx]):
            os.remove(upload_files[client_idx])
        torch.save({'model_state_dict': model.state_dict()},
                   upload_files[client_idx])

        # Save weights periodically
        if np.mod(round_idx + 1, save_interval) == 0:
            torch.save({
                'round': round_idx,
                'sample_idx': sample_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizers[client_idx].state_dict(),
                'scheduler_state_dict': schedulers[client_idx].state_dict(),
                'ssim_log': ssim_log,
                'loss_log': loss_log,
                'coil_log': coil_log,
                'nmse_log': nmse_log,
                'loss': loss,
                'hparams': hparams,
                'train_mask_params': train_mask_params}, local_dir + '/round' +
                str(round_idx) + '_client'+ str(client_idx) + '_before_download.pt')

    # Step all schedulers
    for i in range(num_clients):
        schedulers[i].step()

    ##################FEDERATED WEIGHT SHARING##############################
    with torch.no_grad():
        print("Federated weight averaging occuring at the end of the round")
        # Start with the first client
        saved_model = torch.load(upload_files[0])
        scratch_model.load_state_dict(saved_model['model_state_dict'])
        state_dict_accumulator = copy.deepcopy(scratch_model.state_dict())

        # Accumulate weights from all other clients
        for fed_idx in range(1, num_clients):
            # Only load the model once
            temp_model = torch.load(upload_files[fed_idx])
            scratch_model.load_state_dict(temp_model['model_state_dict'])
            for key in state_dict_accumulator.keys():
                state_dict_accumulator[key] = \
                    state_dict_accumulator[key] + scratch_model.state_dict()[key]
        
        for key in state_dict_accumulator.keys():
            # Average at the end, not sum
            state_dict_accumulator[key] = state_dict_accumulator[key]/num_clients

        # Save ('download') the federated weights in a scratch file
        if os.path.exists(download_file):
            os.remove(download_file)
        # Clients will download this
        torch.save({'model_state_dict': state_dict_accumulator}, download_file)

    # Save and evaluate periodically after downloading
    if np.mod(round_idx + 1, save_interval) == 0:
        # Downloaded weights
        saved_model = torch.load(download_file)
        model.load_state_dict(saved_model['model_state_dict'])

        # Switch to eval
        model.eval()
        running_SSIM_val = 0.0
        plt.figure()

        # !!! For now, hardcoded to one validation set
        with torch.no_grad():
            for sample_idx, sample in tqdm(enumerate(val_loaders[0])):
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
                if sample_idx >=4 and sample_idx < 8:
                    plt.subplot(2, 4, sample_idx-3)
                    plt.imshow(torch.flip(sample['gt_ref_rss'][0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(sample['gt_ref_rss'][0]), cmap='gray')
                    plt.axis('off'); plt.title('GT RSS')
                    plt.subplot(2, 4, sample_idx-3+4*1)
                    plt.imshow(torch.flip(est_crop_rss[0],[0,1]).cpu().detach().numpy(), vmin=0., vmax=torch.max(est_crop_rss[0]), cmap='gray')
                    plt.axis('off'); plt.title('SSIM: %.3f' % (1-ssim_loss.item()))

                # SSIM loss with crop
                ssim_loss = ssim(est_crop_rss[:,None], gt_rss[:,None], data_range)
                running_SSIM_val = running_SSIM_val + (1-ssim_loss.item())

            # Save
            plt.tight_layout()
            plt.savefig(local_dir + '/val_samples_round%d.png' % round_idx, dpi=300)
            plt.close()

            # !!! Hardcoded to validate on the first client site only
            if client_sites[0] <= 3:
                Val_SSIM.append(running_SSIM_val/(num_slices_knee*num_val_pats))
            elif client_sites[0] > 3:
                Val_SSIM.append(running_SSIM_val/(num_slices_brain*num_val_pats))

        # Plot validation metrics
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Training SSIM - Client 0')
        plt.plot(np.asarray(ssim_log[0]))
        plt.subplot(1, 2, 2)
        plt.title('Validation SSIM')
        plt.plot(np.asarray(Val_SSIM))
        # Save
        plt.tight_layout()
        plt.savefig(local_dir + '/train_val_curves_site' + str(client_sites[0]) +
                    '_samples_round%d.png' % round_idx, dpi=300)
        plt.close()

    # Back to train
    model.train()