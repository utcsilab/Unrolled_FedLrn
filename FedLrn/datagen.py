#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py, os
import numpy as np
import sigpy as sp
import torch
from ssdu_masks_gen import ssdu_masks
import torch.fft as torch_fft
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

# Multicoil fastMRI dataset with various options
class MCFullFastMRI(Dataset):
    def __init__(self, sample_list, num_slices,
                 center_slice, downsample, scramble=False,
                 mps_kernel_shape=None, maps=None,
                 direction='y', mask_params=None):
        self.sample_list  = sample_list
        self.num_slices   = num_slices
        self.center_slice = center_slice
        self.downsample   = downsample
        self.mps_kernel_shape = mps_kernel_shape
        self.scramble     = scramble # Scramble samples or not?
        self.maps         = maps # Pre-estimated sensitivity maps
        self.direction    = direction # Which direction are lines in
        self.mask_mode    = mask_params.mask_mode
        
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))
            
        # Masks and related parameters
        self.saved_masks       = [None] * self.__len__() # Single-shot mask storage
        self.saved_theta_masks = [None] * self.__len__() # Single-shot split mask storage
        self.saved_lmbda_masks = [None] * self.__len__()

        # Create mask generator
        if self.mask_mode == 'Gauss':
            self.p_r             = mask_params.p_r # Rho in Mehmet's paper
            self.num_theta_masks = mask_params.num_theta_masks # Lambda sub-splits
            self.theta_fraction  = mask_params.theta_fraction # Fraction of each mask
            self.ssdu_masker     = ssdu_masks(
                self.p_r, num_theta=self.num_theta_masks,
                theta_fraction=self.theta_fraction)
        elif self.mask_mode == 'Accel_only':
            self.num_theta_masks = 1
        else:
            assert False, 'Invalid mask mode in datagen.py!'
            
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
        map_idx = np.mod(idx, self.num_slices) # Maps always count from zero
        
        # Load MRI image
        with h5py.File(self.sample_list[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            k_image = np.asarray(contents['kspace'][slice_idx])
            ref_rss = np.asarray(contents['reconstruction_rss'][slice_idx])
            # Store core file
            core_file  = os.path.basename(self.sample_list[sample_idx])
            core_slice = slice_idx
        
        # If desired, load external sensitivity maps
        if not self.maps is None:
            with h5py.File(self.maps[sample_idx], 'r') as contents:
                # Get sensitivity maps for specific slice
                s_maps      = np.asarray(contents['s_maps'][map_idx])
                s_maps_full = np.copy(s_maps)
        else:
            # Dummy values
            s_maps, s_maps_full = np.asarray([0.]), np.asarray([0.])
            
        # Compute sum-energy of lines
        # !!! This is because some lines are exact zeroes
        line_energy = np.sum(np.square(np.abs(k_image)),
                             axis=(0, 1))
        dead_lines  = np.where(line_energy < 1e-16)[0] # Sufficient based on data
        
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

        # What is the readout direction?
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
            if self.saved_masks[idx] is None:
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
                # Use the corresponding mask
                k_sampling_mask = self.saved_masks[idx]
        else:
            # No downsampling, all ones
            k_sampling_mask = np.ones((k_image.shape[-1],))
        
        # Get measurements
        k_image[..., np.logical_not(k_sampling_mask)] = 0.
        
        # Expand original mask to [nrows, ncol] for input to SSDU function
        if self.direction == 'y':
             k_mask_kx_ky = np.array([k_sampling_mask]*k_image.shape[-2])
        elif self.direction == 'x':
             k_mask_kx_ky = np.array([k_sampling_mask]*k_image.shape[-1]).transpose()
        
        if self.mask_mode == 'Gauss':
            # Only generate these once if not generated before
            if self.saved_theta_masks[idx] is None:
                # Generate Theta (inner, used for CG) and Lambda (outer, used for loss) masks      
                mask_inner_union, mask_outer = \
                    self.ssdu_masker.Gaussian_selection(k_mask_kx_ky.astype(int))
            else:
                # Load pre-existing
                mask_inner_union = self.saved_theta_masks[idx]
                mask_outer       = self.saved_lmbda_masks[idx]
                    
            # Also get union of k-spaces used across unrolls
            k_inner_union = k_image * mask_inner_union
            # Split Theta masks in a set of Sub-Theta masks
            masks_inner = self.ssdu_masker.split_train_masks(mask_inner_union)
        elif self.mask_mode == 'Accel_only':
            # Plain copy
            masks_inner      = [k_sampling_mask] # For consistency
            mask_inner_union = k_sampling_mask
            k_inner_union    = k_image
            mask_outer       = k_sampling_mask
        
        # Get training k-space splits
        if self.direction == 'y':
            k_sampling_mask = k_sampling_mask[None, ...] 
            k_inner = [k_image * masks_inner[idx][None,...]
                       for idx in range(self.num_theta_masks)] # A list of k-space values
            k_outer = k_image * mask_outer[None,...]
        elif self.direction == 'x':
            k_sampling_mask = k_sampling_mask[..., None]
            k_inner = [k_image * masks_inner[idx][None,...]
                       for idx in range(self.num_theta_masks)] # Same as above
            k_outer = k_image * mask_outer[None,...]
        
        # Cast to array
        k_inner     = np.stack(k_inner)
        masks_inner = np.stack(masks_inner)
        
        # Get ACS region
        if self.direction == 'y':
            acs = k_image[..., center_slice_idx.astype(np.int)]
        elif self.direction == 'x':
            acs = k_image[..., center_slice_idx.astype(np.int), :]
        
        # Normalize k-space based on ACS
        max_acs      = np.max(np.abs(acs))
        k_normalized = k_image / max_acs
        k_inner      = k_inner / max_acs
        k_outer      = k_outer / max_acs
        gt_ksp       = gt_ksp / max_acs
        # Scaled GT RSS
        ref_rss      = ref_rss / max_acs
        data_range   = np.max(ref_rss)
        
        # Initial sensitivity maps
        x_coils    = sp.ifft(k_image, axes=(-2, -1))
        x_rss      = np.linalg.norm(x_coils, axis=0, keepdims=True)
        init_maps  = sp.resize(sp.fft(x_coils / x_rss, axes=(-2, -1)),
                               oshape=self.mps_kernel_shape)
        
        sample = {'idx': idx,
                  'ksp': k_normalized.astype(np.complex64),
                  
                  # Partitions of sampled k-space
                  'ksp_inner_union': k_inner_union.astype(np.complex64),
                  'ksp_inner': k_inner.astype(np.complex64),
                  'ksp_outer': k_outer.astype(np.complex64),
                  ###
                  
                  'gt_ksp': gt_ksp.astype(np.complex64),
                  'gt_nonzero_ksp': gt_nonzero_ksp.astype(np.complex64),
                  's_maps': np.stack((np.real(s_maps),
                                      np.imag(s_maps)),
                                     axis=-1).astype(np.float32),
                  's_maps_cplx': s_maps.astype(np.complex64),
                  's_maps_full': s_maps_full.astype(np.complex64),
                  'init_maps': init_maps.astype(np.complex64),
                  'mask': k_sampling_mask.astype(np.float32),
                  
                  # Masks for partitions of sampled k-space
                  'mask_inner_union': mask_inner_union.astype(np.float32),
                  'mask_inner': masks_inner.astype(np.float32),
                  'mask_outer': mask_outer.astype(np.float32),
                  ###
                  
                  'acs_lines': len(center_slice_idx),
                  'dead_lines': dead_lines,
                  'ref_rss': ref_rss.astype(np.float32),
                  'data_range': data_range,
                  'core_file': core_file,
                  'core_slice': core_slice}
        
        return sample
    
    
    
    
# Sense in Image space, exactly like in MoDL
# img -> mult (broad) -> FFT -> mask -> ksp
# ksp -> mask -> IFFT -> mult (conj) -> sum (coils) -> img

    # Forward
def forward_op(img_kernel):
        # Pointwise complex multiply with maps
        mask_fw_ext  = mask[None, ...]
        mult_result = img_kernel[None, ...] * img_y
        
        # Convert back to k-space
        result = torch_fft.ifftshift(mult_result, dim=(-2, -1))
        result = torch_fft.fft2(result, dim=(-2, -1), norm='ortho')
        result = torch_fft.fftshift(result, dim=(-2, -1))
        
        # Multiply with mask
        result = result * mask_fw_ext
        
        return result
    
    # Adjoint
def adjoint_op(ksp,mask,maps):
        # Multiply input with mask and pad
        mask_adj_ext = mask[None, ...]
        ksp_padded  = ksp * mask_adj_ext
        
        # Get image representation of ksp
        img_ksp = torch_fft.fftshift(ksp_padded, dim=(-2, -1))
        img_ksp = torch_fft.ifft2(img_ksp, dim=(-2, -1), norm='ortho')
        img_ksp = torch_fft.ifftshift(img_ksp, dim=(-2, -1))
        
        # Pointwise complex multiply with complex conjugate maps
        mult_result = img_ksp * torch.conj(maps)
        
        # Sum on coil axis
        x_adj = torch.sum(mult_result, dim=0)
        
        return x_adj
    
# Normal operator
def normal_op(img_kernel):
        return adjoint_op(forward_op(img_kernel))
    
# Multicoil fastMRI dataset with various options
class MCFullFastMRIBrain(Dataset):
    def __init__(self, sample_list, num_slices,
                 center_slice, downsample, scramble=False,
                 mps_kernel_shape=None, maps=None,
                 direction='y', mask_params=None):
        self.sample_list  = sample_list
        self.num_slices   = num_slices
        self.center_slice = center_slice
        self.downsample   = downsample
        self.mps_kernel_shape = mps_kernel_shape
        self.scramble     = scramble # Scramble samples or not?
        self.maps         = maps # Pre-estimated sensitivity maps
        self.direction    = direction # Which direction are lines in
        self.mask_mode    = mask_params.mask_mode
        
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))
            
        # Masks and related parameters
        self.saved_masks       = [None] * self.__len__() # Single-shot mask storage
        self.saved_theta_masks = [None] * self.__len__() # Single-shot split mask storage
        self.saved_lmbda_masks = [None] * self.__len__()

        # Create mask generator
        if self.mask_mode == 'Gauss':
            self.p_r             = mask_params.p_r # Rho in Mehmet's paper
            self.num_theta_masks = mask_params.num_theta_masks # Lambda sub-splits
            self.theta_fraction  = mask_params.theta_fraction # Fraction of each mask
            self.ssdu_masker     = ssdu_masks(
                self.p_r, num_theta=self.num_theta_masks,
                theta_fraction=self.theta_fraction)
        elif self.mask_mode == 'Accel_only':
            self.num_theta_masks = 1
        else:
            assert False, 'Invalid mask mode in datagen.py!'
            
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
        map_idx = np.mod(idx, self.num_slices) # Maps always count from zero
        
        # Load MRI image
        with h5py.File(self.sample_list[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            k_image = np.asarray(contents['kspace'][slice_idx])
            ref_rss = np.asarray(contents['reconstruction_rss'][slice_idx])
            # Store core file
            core_file  = os.path.basename(self.sample_list[sample_idx])
            core_slice = slice_idx
        
        # If desired, load external sensitivity maps
#         sample_idx=0
#         print(sample_idx)
        if not self.maps is None:
            with h5py.File(self.maps[sample_idx], 'r') as contents:
                # Get sensitivity maps for specific slice
                s_maps      = np.asarray(contents['s_maps'][slice_idx])#was map_idx
                s_maps_full = np.copy(s_maps)
        else:
            # Dummy values
            s_maps, s_maps_full = np.asarray([0.]), np.asarray([0.])
            
        # Compute sum-energy of lines
        # !!! This is because some lines are exact zeroes
        line_energy = np.sum(np.square(np.abs(k_image)),
                             axis=(0, 1))
        dead_lines  = np.where(line_energy < 1e-16)[0] # Sufficient based on data
        
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
      #  k_image = np.delete(k_image, dead_lines, axis=-1)
        # Remove them from the frequency representation of the maps as well

#        if not self.maps is None:
 #           k_maps = sp.fft(s_maps, axes=(-2, -1))
      #      k_maps = np.delete(k_maps, dead_lines, axis=-1)
  #          s_maps = sp.ifft(k_maps, axes=(-2, -1))
    
        # Store GT data without zero lines
        gt_nonzero_ksp = np.copy(k_image)

        # What is the readout direction?
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
            if self.saved_masks[idx] is None:
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
                # Use the corresponding mask
                k_sampling_mask = self.saved_masks[idx]
        else:
            # No downsampling, all ones
            k_sampling_mask = np.ones((k_image.shape[-1],))
        
        # Get measurements
        k_image[..., np.logical_not(k_sampling_mask)] = 0.
        
        # Expand original mask to [nrows, ncol] for input to SSDU function
        if self.direction == 'y':
             k_mask_kx_ky = np.array([k_sampling_mask]*k_image.shape[-2])
        elif self.direction == 'x':
             k_mask_kx_ky = np.array([k_sampling_mask]*k_image.shape[-1]).transpose()
        
        if self.mask_mode == 'Gauss':
            # Only generate these once if not generated before
            if self.saved_theta_masks[idx] is None:
                # Generate Theta (inner, used for CG) and Lambda (outer, used for loss) masks      
                mask_inner_union, mask_outer = \
                    self.ssdu_masker.Gaussian_selection(k_mask_kx_ky.astype(int))
            else:
                # Load pre-existing
                mask_inner_union = self.saved_theta_masks[idx]
                mask_outer       = self.saved_lmbda_masks[idx]
                    
            # Also get union of k-spaces used across unrolls
            k_inner_union = k_image * mask_inner_union
            # Split Theta masks in a set of Sub-Theta masks
            masks_inner = self.ssdu_masker.split_train_masks(mask_inner_union)
        elif self.mask_mode == 'Accel_only':
            # Plain copy
            masks_inner      = [k_sampling_mask] # For consistency
            mask_inner_union = k_sampling_mask
            k_inner_union    = k_image
            mask_outer       = k_sampling_mask
        
        # Get training k-space splits
        if self.direction == 'y':
            k_sampling_mask = k_sampling_mask[None, ...] 
            k_inner = [k_image * masks_inner[idx][None,...]
                       for idx in range(self.num_theta_masks)] # A list of k-space values
            k_outer = k_image * mask_outer[None,...]
        elif self.direction == 'x':
            k_sampling_mask = k_sampling_mask[..., None]
            k_inner = [k_image * masks_inner[idx][None,...]
                       for idx in range(self.num_theta_masks)] # Same as above
            k_outer = k_image * mask_outer[None,...]
        
        # Cast to array
        k_inner     = np.stack(k_inner)
        masks_inner = np.stack(masks_inner)
        
        # Get ACS region
        if self.direction == 'y':
            acs = k_image[..., center_slice_idx.astype(np.int)]
        elif self.direction == 'x':
            acs = k_image[..., center_slice_idx.astype(np.int), :]
        
        # Normalize k-space based on ACS
        max_acs      = np.max(np.abs(acs))
        k_normalized = k_image / max_acs
        k_inner      = k_inner / max_acs
        k_outer      = k_outer / max_acs
        gt_ksp       = gt_ksp / max_acs
        # Scaled GT RSS
        ref_rss      = ref_rss / max_acs
        data_range   = np.max(ref_rss)
        
        # Initial sensitivity maps
        x_coils    = sp.ifft(k_image, axes=(-2, -1))
        x_rss      = np.linalg.norm(x_coils, axis=0, keepdims=True)
        init_maps  = sp.resize(sp.fft(x_coils / x_rss, axes=(-2, -1)),
                               oshape=self.mps_kernel_shape)
        
        sample = {'idx': idx,
                  'ksp': k_normalized.astype(np.complex64),
                  
                  # Partitions of sampled k-space
                  'ksp_inner_union': k_inner_union.astype(np.complex64),
                  'ksp_inner': k_inner.astype(np.complex64),
                  'ksp_outer': k_outer.astype(np.complex64),
                  ###
                  
                  'gt_ksp': gt_ksp.astype(np.complex64),
                  'gt_nonzero_ksp': gt_nonzero_ksp.astype(np.complex64),
                  's_maps': np.stack((np.real(s_maps),
                                      np.imag(s_maps)),
                                     axis=-1).astype(np.float32),
                  's_maps_cplx': s_maps_full.astype(np.complex64),
                  's_maps_full': s_maps_full.astype(np.complex64),
                  'init_maps': init_maps.astype(np.complex64),
                  'mask': k_sampling_mask.astype(np.float32),
                  
                  # Masks for partitions of sampled k-space
                  'mask_inner_union': mask_inner_union.astype(np.float32),
                  'mask_inner': masks_inner.astype(np.float32),
                  'mask_outer': mask_outer.astype(np.float32),
                  ###
                  
                  'acs_lines': len(center_slice_idx),
                  'dead_lines': dead_lines,
                  'ref_rss': ref_rss.astype(np.float32),
                  'data_range': data_range,
                  'core_file': core_file,
                  'core_slice': core_slice}
        
        return sample
