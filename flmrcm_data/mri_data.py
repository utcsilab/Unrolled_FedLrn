"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib, copy
import random

from tqdm import tqdm

import numpy as np
import h5py
from torch.utils.data import Dataset
from flmrcm_data import transforms
import torch

class SliceDataPreloaded(Dataset):
    def __init__(self, sample_list, transform, num_slices, center_slice,
                 downsample, scramble=False, preload=True):
        self.sample_list      = sample_list
        self.num_slices       = num_slices
        self.center_slice     = center_slice
        self.downsample       = downsample
        self.scramble         = scramble # Scramble samples or not?
        self.preload          = preload
        self.transform        = transform
        
        if self.preload:
            print('Preloading files!')
            self.preloaded_slices = []
            self.preloaded_rss = []
            
            # For each patient
            for patient_idx in tqdm(range(len(sample_list))):
                # A local list
                local_slices = []
                local_rss    = []
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
                    
                    # Save
                    local_slices.append(k_image)
                    local_rss.append(ref_rss)
                    
                # Convert to arrays - always tileable
                local_slices = np.asarray(local_slices)
                local_rss    = np.asarray(local_rss)
                
                # Add to global list - not generally tileable
                self.preloaded_slices.append(copy.deepcopy(local_slices))
                self.preloaded_rss.append(copy.deepcopy(local_rss))
                
            print('Finished preloading!')
        
        if self.scramble:
            # One time permutation
            self.permute = np.random.permutation(self.__len__())
            self.inv_permute = np.zeros(self.permute.shape)
            self.inv_permute[self.permute] = np.arange(len(self.permute))

    def __len__(self):
        return int(len(self.sample_list) * self.num_slices)
    
    def __getitem__(self, idx):
        # Permute if desired
        if self.scramble:
            idx = self.permute[idx]
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract from preloaded collections
        if self.preload:
            # Separate slice (counting from zero!) and sample
            sample_idx = idx // self.num_slices
            slice_idx  = np.mod(idx, self.num_slices)
            
            k_image = copy.deepcopy(self.preloaded_slices[sample_idx][slice_idx])
            ref_rss = self.preloaded_rss[sample_idx][slice_idx]
        
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

        # Remove dead lines completely
        k_image = np.delete(k_image, dead_lines, axis=-1)
        # What is the readout direction?site
        sampling_axis = -1

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
            mask = np.isin(np.arange(k_image.shape[sampling_axis]),
                                      np.hstack((center_slice_idx,
                                                 random_line_idx)))
        else:
            # No downsampling, all ones
            mask = np.ones((k_image.shape[-1],))
            
        # Inflate mask dimensions
        mask = mask[None, None, :, None]
        
        return self.transform(k_image, mask, ref_rss, 
                  np.asarray([0.]), np.asarray([0.]), np.asarray([0.]))
    


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, resolution, which_challenge, mask_func=None, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        """
        kspace     = transforms.to_tensor(kspace)
        resolution = target.shape[-1]
        
        # Apply mask
        masked_kspace = kspace * mask

        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        
        # Crop input image to given resolution if larger
        smallest_width = min(resolution, image.shape[-2])
        smallest_height = min(resolution, image.shape[-3])
        if target is not None:
            smallest_width = min(smallest_width, target.shape[-1])
            smallest_height = min(smallest_height, target.shape[-2])

        crop_size = (smallest_height, smallest_width)
        image = transforms.complex_center_crop(image, crop_size)
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        # Normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])
        return image, target, mean, std, 0, 0, 0

