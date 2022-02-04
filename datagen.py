#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, os, copy
import numpy as np
import sigpy as sp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def crop(variable, tw, th):
    w, h = variable.shape[-2:]
    x1 = int(np.ceil((w - tw) / 2.))
    y1 = int(np.ceil((h - th) / 2.))
    return variable[..., x1:x1+tw, y1:y1+th]

def crop_cplx(variable, tw, th):
    w, h = variable.shape[-3:-1]
    x1 = int(np.ceil((w - tw) / 2.))
    y1 = int(np.ceil((h - th) / 2.))
    return variable[..., x1:x1+tw, y1:y1+th, :]

def RSS_recon(ksp):
    coil_image = sp.ifft(ksp, axes = (-2,-1))
    image = np.sum(abs(coil_image)**2, axis = 0)
    image = np.sqrt(image)
    return image

# Multicoil fastMRI dataset with various options(Previously called MCFullFastMRIBrain)
class MCFullFastMRI(Dataset):
    def __init__(self, sample_list, num_slices,
                 center_slice, downsample, scramble=False,
                 mps_kernel_shape=None, maps=None,
                 sites=None, direction='y',
                 mask_params=None, noise_stdev=0.0,
                 preload=True):
        self.sample_list      = sample_list
        self.num_slices       = num_slices
        self.center_slice     = center_slice
        self.downsample       = downsample
        self.mps_kernel_shape = mps_kernel_shape
        self.scramble         = scramble # Scramble samples or not?
        self.maps             = maps # Pre-estimated sensitivity maps
        self.direction        = direction # Which direction are lines in
        self.stdev            = noise_stdev
        self.sites            = sites
        self.preload          = preload
        
        if self.preload:
            print('Preloading files!')
            self.preloaded_slices = []
            self.preloaded_maps, self.preloaded_rss = [], []
            
            # For each patient
            for patient_idx in tqdm(range(len(sample_list))):
                # A local list
                local_slices, local_mps = [], []
                local_rss               = []
                # For each slice - offset from center
                for slice_idx in range(num_slices):
                    true_slice_idx = \
                        self.center_slice + slice_idx - self.num_slices // 2
                    
                    # Load measurements
                    with h5py.File(self.sample_list[patient_idx], 'r') as contents:
                        # Get k-space for specific slice
                        k_image = np.asarray(contents['kspace'][true_slice_idx])
                        ref_rss = np.asarray(
                            contents['reconstruction_rss'][true_slice_idx])
                    # Load maps
                    with h5py.File(self.maps[patient_idx], 'r') as contents:
                        s_maps = np.asarray(contents['s_maps'][true_slice_idx])
                    
                    # Save
                    local_slices.append(k_image)
                    local_rss.append(ref_rss)
                    local_mps.append(s_maps)                      
                    
                # Convert to arrays - always tileable
                local_slices = np.asarray(local_slices)
                local_mps    = np.asarray(local_mps)
                local_rss    = np.asarray(local_rss)
                assert np.all(
                    local_slices.shape == local_mps.shape), 'Incorrect maps!'
                
                # Add to global list - not generally tileable
                self.preloaded_slices.append(copy.deepcopy(local_slices))
                self.preloaded_maps.append(copy.deepcopy(local_mps))
                self.preloaded_rss.append(copy.deepcopy(local_rss))
                
            print('Finished preloading!')
        
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))

        # Deprecated
        self.num_theta_masks = 1

    def __len__(self):
        return len(self.sample_list) * self.num_slices

    def __getitem__(self, idx):
        # Permute if desired
        if self.scramble:
            idx = self.permute[idx]

        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + \
            np.mod(idx, self.num_slices) - self.num_slices // 2

        # Extract from preloaded collections
        if self.preload:
            k_image = self.preloaded_slices[sample_idx][slice_idx]
            ref_rss = self.preloaded_rss[sample_idx][slice_idx]
            s_maps  = self.preloaded_maps[sample_idx][slice_idx]
        else:
            # Load MRI samples and maps
            with h5py.File(self.sample_list[sample_idx], 'r') as contents:
                # Get k-space for specific slice
                k_image = np.asarray(contents['kspace'][slice_idx])
                ref_rss = np.asarray(contents['reconstruction_rss'][slice_idx])
                
            # If desired, load external sensitivity maps
            if not self.maps is None:
                with h5py.File(self.maps[sample_idx], 'r') as contents:
                    # Get sensitivity maps for specific slice
                    s_maps = np.asarray(contents['s_maps'][slice_idx])#was map_idx
            else:
                # Dummy values
                s_maps = np.asarray([0.])
                
        # Store core file
        core_file  = os.path.basename(self.sample_list[sample_idx])
        core_slice = slice_idx

        # Assign sites
        if self.sites is None:
            site = 0 # Always the first one for a client
        else:
            site = self.sites[sample_idx] - 1 # Counting indices from zero

        # Compute sum-energy of lines
        # !!! This is because some lines are exact zeroes
        line_energy = np.sum(np.square(np.abs(k_image)),
                             axis=(0, 1))
        dead_lines  = np.where(line_energy < 1e-30)[0] # Sufficient based on data

        # Always remove an even number of lines to keep original centering
        dead_lines_front = np.sum(dead_lines < 160)
        dead_lines_back  = np.sum(dead_lines > 160)
        if np.mod(dead_lines_front, 2):
            dead_lines = np.delete(dead_lines, 0)
        if np.mod(dead_lines_back, 2):
            dead_lines = np.delete(dead_lines, -1)

        # Store all GT data
        gt_ksp = np.copy(k_image)

        # Remove dead lines completely
        k_image = np.delete(k_image, dead_lines, axis=-1)
        # Remove them from the frequency representation of the maps as well
        if not self.maps is None:
            k_maps = sp.fft(s_maps, axes=(-2, -1))
            k_maps = np.delete(k_maps, dead_lines, axis=-1)
            s_maps = sp.ifft(k_maps, axes=(-2, -1))

        # Store GT data without zero lines
        gt_nonzero_ksp = np.copy(k_image)

        # What is the readout direction?site
        sampling_axis = -1 if self.direction == 'y' else -2

        # Fixed percentage of central lines, as per fastMRI
        if self.downsample >= 2 and self.downsample <= 6:
            num_central_lines = np.round(0.08 * k_image.shape[sampling_axis])
        else:
            num_central_lines = np.round(0.04 * k_image.shape[sampling_axis])
        center_slice_idx = np.arange(
            (k_image.shape[sampling_axis] - num_central_lines) // 2,
            (k_image.shape[sampling_axis] + num_central_lines) // 2)

        # Downsampling
        # !!! Unlike fastMRI, we always pick lines to ensure R = downsample
        if self.downsample > 1.01:
            # Candidates outside the central region
            random_line_candidates = np.setdiff1d(np.arange(
                k_image.shape[sampling_axis]), center_slice_idx)

            # If masks are not fed, generate them on the spot
            # Pick random lines outside the center location
            random_line_idx = np.random.choice(
                random_line_candidates,
                size=(int(k_image.shape[sampling_axis] // self.downsample) -
                      len(center_slice_idx)),
                replace=False)

            # Create sampling mask and downsampled k-space data
            k_sampling_mask = np.isin(np.arange(k_image.shape[sampling_axis]),
                                      np.hstack((center_slice_idx,
                                                 random_line_idx)))
        else:
            # No downsampling, all ones
            k_sampling_mask = np.ones((k_image.shape[-1],))

        # Get normalization constant from undersampled RSS
        acs_image = sp.rss(sp.ifft(sp.resize(sp.resize(k_image, 
               [k_image.shape[0], int(num_central_lines),
                int(num_central_lines)]), gt_ksp.shape), 
                                   axes=(-1,-2)), axes=(0,))
        norm_const = np.percentile(acs_image, 99)

        # Optional: Add noise to ksp
        if self.stdev > 1e-20:
            noise   = np.random.randn(*k_image.shape) + \
                1j * np.random.randn(*k_image.shape)
            k_image = k_image + 1 / np.sqrt(2) * self.stdev * noise * norm_const

        # Get measurements
        k_image[..., np.logical_not(k_sampling_mask)] = 0.

        # Normalize k-space based on ACS
        k_normalized   = k_image / norm_const
        gt_ksp         = gt_ksp / norm_const
        gt_nonzero_ksp = gt_nonzero_ksp / norm_const
        
        # Scaled GT RSS
        gt_nonzero_ksp_pad = sp.resize(gt_nonzero_ksp, gt_ksp.shape)
        gt_ref_rss   = sp.resize(sp.rss(sp.ifft(gt_nonzero_ksp_pad,
                            axes=(-1,-2)), axes=(0,)), ref_rss.shape)
        ref_rss      = ref_rss / norm_const
        data_range   = np.max(gt_ref_rss)
        
        sample = {'idx': idx,
                  'ksp': k_normalized.astype(np.complex64),
                  'site_idx': int(site),
                  'acs_image': acs_image.astype(np.float32),
                  'norm_const': norm_const.astype(np.float32),
                  'gt_ksp': gt_ksp.astype(np.complex64),
                  's_maps_cplx': s_maps.astype(np.complex64),
                  'mask': k_sampling_mask.astype(np.float32),
                  'acs_lines': len(center_slice_idx),
                  'dead_lines': dead_lines,
                  'gt_ref_rss': gt_ref_rss.astype(np.float32),
                  'ref_rss': ref_rss.astype(np.float32),
                  'data_range': data_range,
                  'core_file': core_file,
                  'core_slice': core_slice}

        return sample