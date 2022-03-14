#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, re
sys.path.insert(0, '.')

import torch, os, copy
import numpy as np
from dotmap import DotMap

from losses import SSIMLoss, MCLoss, NMSELoss
from matplotlib import pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from site_loader import site_loader

from flmrcm_models.unet_model import UnetModel_ad_da, Feature_discriminator
from flmrcm_models.recon_Update import LocalUpdate_ad_da
from flmrcm_models.Fed import FedAvg
from flmrcm_data.mri_data import DataTransform, SliceDataPreloaded
from tensorboardX import SummaryWriter

plt.rcParams.update({'font.size': 12})
plt.ioff(); plt.close('all')

# Maybe
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
# Always !!!
torch.backends.cudnn.benchmark = True

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--seed'    , type=int, default=1500       , help='random seed to use')
    parser.add_argument('--mini_seed', type=int, default=0         , help='an extra seed to shuffle clients')
    parser.add_argument('--GPU'     , type=int, default=0          , help='GPU to Use')
    parser.add_argument('--num_work', type=int, default=2          , help='number of workers to use')
    parser.add_argument('--train_dilation', type=int, default=100  , help='boost num rounds, save interval and decay in 24:1:16 ratio')
    parser.add_argument('--ch'      , type=int, default=16         , help='number of channels in Unet')
    parser.add_argument('--num_pool', type=int, default=3          , help='number of pool layer in UNet')
    parser.add_argument('--lr'      , type=float, default=1e-4     , help='learning rate for training')
    parser.add_argument('--client_pats' , nargs='+', default=[1, 1, 1], type=int, help='Vector of client samples (patients, 10 slices each)')
    parser.add_argument('--client_sites', nargs='+', default=[4, 11, 8], type=int, help='Vector of client sites (local data distribution)')
    parser.add_argument('--target_client_idx', type=int, default=1         , help='Who is the target client?')
    parser.add_argument('--share_int', type= int, default=2, help='how often do we share weights(measured in epochs)')

    args = parser.parse_args()
    return args

# Get arguments
args = get_args()
print(args)

GPU_ID                = args.GPU
global_seed           = args.seed
num_workers           = args.num_work
lr                    = args.lr
unet_ch               = args.ch
unet_num_pool         = args.num_pool

# Follow the ratio 48:1:32
num_rounds            = args.train_dilation * 48
save_interval         = args.train_dilation * 1
decay_epochs          = args.train_dilation * 32

# Federation stuff
share_int             = args.share_int
client_pats           = args.client_pats
client_sites          = args.client_sites
target_client_idx     = args.target_client_idx

# More federation stuff
num_val_pats          = 1

# Immediately determine number of clients
assert len(client_pats) == len(client_sites), 'Client args mismatch!'
num_clients = len(client_pats)
client_perm = np.arange(num_clients)

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

##################### Model config###########################################
hparams                 = DotMap()
hparams.mode            = 'MoDL'
hparams.logging         = False
# Mode-specific settings
if hparams.mode == 'MoDL':
    hparams.use_map_net     = False # No Map-net
    hparams.map_init        = 'espirit'
    hparams.block1_max_iter = 0 # Maps

hparams.img_arch     = 'UNet' # 'UNet' only
hparams.img_channels = unet_ch
hparams.img_blocks   = unet_num_pool
if hparams.img_arch != 'UNet':
    hparams.latent_channels = 64
hparams.downsample   = 4 # Training R
hparams.kernel_size  = 3
hparams.in_channels  = 2
hparams.share_mode   = 'Seperate' # either Seperate, Full_avg or Split_avg for how we want to share weights
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
#################Set save location directory################################
# Auxiliary strings
site_string = '_'.join([str(client_sites[idx]) for idx in range(num_clients)])
pats_string = '_'.join([str(client_pats[idx]) for idx in range(num_clients)])

global_dir = 'Results/federated/clients%d_sites%s/\
FL-MRCM_target%d/sync%d/UNet_pool%d_ch%d/train_pats_%s/seed%d' % (
      num_clients, site_string, target_client_idx, share_int,
      unet_num_pool, unet_ch, pats_string, global_seed)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
log_dir = global_dir + '/summary'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Get a core model
net_glob = UnetModel_ad_da(in_chans=1, out_chans=1,
    chans=unet_ch, num_pool_layers=unet_num_pool, drop_prob=0.0).cuda()
net_glob.train()
# Count parameters
total_params = np.sum([np.prod(p.shape) for p
                       in net_glob.parameters() if p.requires_grad])
print('Total parameters %d' % total_params)

# Get models per client
G_s, FD = [], []
for i in range(num_clients):
    G_s.append(UnetModel_ad_da(
        in_chans=1, out_chans=1, chans=unet_ch,
        num_pool_layers=unet_num_pool, drop_prob=0.0).cuda())
    FD.append(Feature_discriminator(input_dim=512).cuda())
    
# Get optimizers per client
opt_g_s, opt_FD = [], []
for i in range(num_clients):
    opt_g_s.append(torch.optim.RMSprop(G_s[i].parameters(), lr=args.lr))
    opt_FD.append(torch.optim.RMSprop(FD[i].parameters(), lr=args.lr*10))
    
# Initialize weights
w_glob = net_glob.state_dict()
for G in G_s:
    for net, net_cardinal in zip(G.named_parameters(), net_glob.named_parameters()):
        net[1].data = net_cardinal[1].data.clone()

# Get transform functions
train_data_transform = DataTransform(None, 'multicoil', None, use_seed=False)
val_data_transform = DataTransform(None, 'multicoil', None)

######## Get client datasets and dataloaders ############
cursors   = [] # make a cursor for each unique site(used (not client)
unique_sites = list(set(client_sites))
num_unique = len(unique_sites)
print('unique sites:', unique_sites)
print('# of unique sites:', num_unique)
for i in range(num_unique):
    cursors.append(0)

# Get unique data for each source client
global_client_files = []
for client_idx in range(num_clients):
    local_file = '/run/user/1000/gvfs/sftp:host=axiom.ece.utexas.edu,\
port=51022,user=marius/fast/marius/federated/newest_results_jan14/\
clientAdam_clients10_sites1_2_4_5_6_7_8_9_10_11/personalLam0/Scaffold/\
clientLR6.0e-04_clientAdam/sync100/UNet_pool3_ch16/\
train_pats_5_5_5_5_5_5_5_5_5_5/seed1500/N6_n0_lamInit0.100/\
round4_client%d_before_download.pt' % client_idx
    contents = torch.load(local_file)
    global_client_files.append(contents['client_files'][:client_pats[client_idx]])

global_train_datasets, global_val_datasets = [], []
for i in client_perm:
    # Overwrite client site
    integers = re.findall(r'\d+', global_client_files[i][0])
    client_sites[i] = int(integers[0])
    
    # Get all the filenames from the site
    _, _, all_val_files, all_val_maps = \
        site_loader(client_sites[i], 100, num_val_pats)#always load the maximum number of patients for that site then parse it down

    unique_site_idx = unique_sites.index(client_sites[i]) #get site index for cursor
    print('unique site idx:', unique_site_idx)
    # Subselect unique (by incrementing) training samples
    client_idx = np.arange(cursors[unique_site_idx], 
           cursors[unique_site_idx] + client_pats[i]).astype(int)
    print('client_idx:', client_idx)
    client_train_files = global_client_files[i]

    # Increment cursor for the site
    cursors[unique_site_idx] = cursors[unique_site_idx] + client_pats[i]

    # Maintenance
    if client_sites[i] <= 4: # must be a knee site
        center_slice = center_slice_knee
        num_slices   = num_slices_knee
    elif client_sites[i] > 4: # must be a brain site
        center_slice = center_slice_brain
        num_slices   = num_slices_brain

    # Create training client dataset
    train_source_dataset = SliceDataPreloaded(client_train_files,
              train_data_transform, num_slices, center_slice, hparams.downsample)
    # Create validation client dataset
    val_source_dataset   = SliceDataPreloaded(all_val_files, 
              val_data_transform, num_slices, center_slice, hparams.downsample,
              scramble=True)
    global_train_datasets.append(copy.deepcopy(train_source_dataset))
    global_val_datasets.append(copy.deepcopy(val_source_dataset))
    
# Split datasets in source-target
source_idx           = np.setdiff1d(np.arange(num_clients), target_client_idx)
target_idx           = np.setdiff1d(np.arange(num_clients), source_idx)
datasets_list        = [global_train_datasets[idx] for idx in np.arange(num_clients)]
target_datasets_list = [global_train_datasets[target_idx[0]] for _ in range(num_clients)]
dataset_val          = global_val_datasets[target_idx[0]]
val_loader           = DataLoader(dataset_val, batch_size=1, num_workers=4, shuffle=False)

# Criterions
ssim           = SSIMLoss().cuda()
multicoil_loss = MCLoss().cuda()
pixel_loss     = torch.nn.MSELoss(reduction='sum')
nmse_loss      = NMSELoss()

# Logs
Val_SSIM, ssim_loss = [], [] # !!! Only supports one 'client' = downloaded model
training_logs = {'L1': [], 'total_loss': [],
                 'adv_g': [], 'adv_d': []}

# Inner directory
local_dir = global_dir

# Training loop
start_epoch = -1
for iter in tqdm(range(start_epoch+1, num_rounds)):
    # Safety
    net_glob.train()
    w_locals, loss_locals = [], []
    
    # For each client update local models
    for idx, dataset_train in enumerate(datasets_list):
        flag = datasets_list[idx] == target_datasets_list[0] # for disable adv loss for target dataset
        local = LocalUpdate_ad_da(args=args, device='cuda:0', dataset=dataset_train,
                               dataset_target = target_datasets_list[idx],
                               optimizer=opt_g_s[idx],optimizer_fd=opt_FD[idx], flag=flag)
        # models communication
        G_s[idx].load_state_dict(net_glob.state_dict())
        # global update
        w, loss = local.train(net=G_s[idx], net_fd=FD[idx], epoch=iter, idx=idx, writer=writer,
                              logs=training_logs)
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        
        # Save incrementally
        torch.save({'training_logs': training_logs}, local_dir + '/incremental_logs.pt')
        
    # update global weights
    w_glob = FedAvg(w_locals)
    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)
    
    # print loss
    loss_avg = np.sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

    # Save networks
    if np.mod(iter + 1, save_interval) == 0:
        # Save federated weights
        torch.save({'model_state_dict': net_glob.state_dict(),
                    'training_logs': training_logs},
                   local_dir + '/round' + str(iter) + '_federated.pt')
        # Save clients before federation
        for client_idx in range(num_clients):
            torch.save({'model_state_dict': G_s[idx].state_dict()},
                       local_dir + '/round' + \
str(iter) + '_clientG'+ str(client_idx) + '_before_download.pt')
            torch.save({'model_state_dict': FD[idx].state_dict()},
                       local_dir + '/round' + \
str(iter) + '_clientD'+ str(client_idx) + '_before_download.pt')
    
    # Validation
    if np.mod(iter + 1, save_interval) == 0:
        with torch.no_grad():
            # Switch to eval
            net_glob.eval()
            val_ssim = []
            plt.figure()
            
            for sample_idx, sample in tqdm(enumerate(val_loader)):
                # Move to CUDA
                input, target, mean, std, _, _ = sample
                input  = input.cuda()
                target = target.cuda()
    
                # Get outputs
                est_rss_cropped = net_glob(input)
    
                # Re-scale RSS back to natural intervals
                est_rss_cropped = est_rss_cropped * std + mean
                target          = target * std + mean
    
                # Get vmax
                vmax = torch.max(target)
                
                # SSIM
                ssim_loss = ssim(input[:, None], target[:, None], vmax)
    
                # Add the first few ones to plot
                if sample_idx >=4 and sample_idx < 8:
                    plt.subplot(2, 4, sample_idx-3)
                    plt.imshow(torch.flip(target).cpu().detach().numpy(), vmin=0., 
                               vmax=vmax, cmap='gray')
                    plt.axis('off'); plt.title('GT RSS')
                    plt.subplot(2, 4, sample_idx-3+4*1)
                    plt.imshow(torch.flip(est_rss_cropped).cpu().detach().numpy(),
                               vmin=0., vmax=vmax, cmap='gray')
                    plt.axis('off'); plt.title('SSIM: %.3f' % (1-ssim_loss.item()))
    
                # Log SSIM
                val_ssim.append(1-ssim_loss.item())
    
            # Save
            plt.tight_layout()
            plt.savefig(local_dir + '/val_samples_round%d.png' % iter, dpi=300)
            plt.close()
    
        # Save validation metrics
        torch.save({'val_ssim': np.asarray(val_ssim)}, # Federated
            local_dir + '/inference_target_client_round%d.pt' % iter)
        
writer.close()