#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import sigpy as sp
import numpy as np
import copy as copy

from core_ops import TorchMoDLSense, TorchMoDLImage

from opt import ZConjGrad
from unet import NormUnet

# Unrolled J-Sense in MoDL style
class MoDLDoubleUnroll(torch.nn.Module):
    def __init__(self, hparams):
        super(MoDLDoubleUnroll, self).__init__()
        # Storage
        self.verbose    = hparams.verbose
        self.batch_size = hparams.batch_size
        self.block1_max_iter = hparams.block1_max_iter
        self.block2_max_iter = hparams.block2_max_iter
        self.cg_eps          = hparams.cg_eps

        # Modes
        self.mode        = hparams.mode
        self.use_img_net = hparams.use_img_net
        self.use_map_net = hparams.use_map_net
        self.num_theta_masks = hparams.num_theta_masks
        self.mask_mode       = hparams.mask_mode
        # Map modes
        self.map_mode = hparams.map_mode
        self.map_norm = hparams.map_norm
        # Initial variables
        self.map_init = hparams.map_init
        self.img_init = hparams.img_init
        # Logging
        self.logging  = hparams.logging

        # ImageNet parameters
        self.img_channels = hparams.img_channels
        self.img_blocks   = hparams.img_blocks
        self.img_arch     = hparams.img_arch
        # the following parameters are only useful for Federated Learning
        self.img_sep      = hparams.img_sep
        self.all_sep      = hparams.all_sep
        self.share_global = hparams.share_global
        self.num_blocks1     = hparams.num_blocks1
        self.num_blocks2     = hparams.num_blocks2
        # MapNet parameters
        self.map_channels = hparams.map_channels
        self.map_blocks   = hparams.map_blocks
        # Attention parameters
        self.att_config   = hparams.att_config
        if hparams.img_arch != 'Unet':
            self.latent_channels = hparams.latent_channels
            self.kernel_size     = hparams.kernel_size
        # Size parameters
        self.mps_kernel_shape = hparams.mps_kernel_shape # B x C x h x w
        # Get useful values
        self.num_coils = self.mps_kernel_shape[-3]
        self.ones_mask = torch.ones((1)).cuda()

        # Initialize trainable parameters
        if hparams.l2lam_train:
            self.block1_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
            self.block2_l2lam = torch.nn.Parameter(torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda())
        else:
            self.block1_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()
            self.block2_l2lam = torch.tensor(
                    hparams.l2lam_init *
                    np.ones((1))).cuda()

        # Initialize image module
        if hparams.use_img_net:
            # Initialize ResNet module
            if self.img_sep: # Do we use separate networks at each unroll?
                assert False, 'Deprecated!'
            else:
                if self.img_arch == 'ResNet':
                    assert False, 'Deprecated!'
                elif self.img_arch == 'UNet':
                    # !!! Normalizes
                    self.image_net = NormUnet(chans=self.img_channels,
                                              num_pools=self.img_blocks)
                elif self.img_arch == 'ResNetSplit':
                    assert False, 'Deprecated!'
        else:
            # Bypass
            self.image_net = torch.nn.Identity()

        # Initialize map module
        if hparams.use_map_net:
            assert False, 'Deprecated!'
        else:
            # Bypass
            self.maps_net  = torch.nn.Identity()

        # Initial 'fixed' maps
        # See in 'forward' for the exact initialization depending on mode
        self.init_maps_kernel = 0. * torch.randn((self.batch_size,) +
                      tuple(self.mps_kernel_shape) + (2,)).cuda()
        self.init_maps_kernel = torch.view_as_complex(self.init_maps_kernel)

    # Get torch operators for the entire batch
    def get_core_torch_ops(self, mps_kernel, img_kernel, mask, direction):
        # List of output ops
        normal_ops, adjoint_ops, forward_ops = [], [], []

        # For each sample in batch
        for idx in range(self.batch_size):
            # Type
            if direction == 'ConvSense':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLSense(mps_kernel[idx], mask[idx])
            elif direction == 'ConvImage':
                forward_op, adjoint_op, normal_op = \
                    TorchMoDLImage(img_kernel[idx], mask[idx])

            # Add to lists
            normal_ops.append(normal_op)
            adjoint_ops.append(adjoint_op)
            forward_ops.append(forward_op)

        # Return operators
        return normal_ops, adjoint_ops, forward_ops

    # Given a batch of inputs and ops, get a single batch operator
    def get_batch_op(self, input_ops, batch_size):
        # Inner function trick
        def core_function(x):
            # Store in list
            output_list = []
            for idx in range(batch_size):
                output_list.append(input_ops[idx](x[idx])[None, ...])
            # Stack and return
            return torch.cat(output_list, dim=0)
        return core_function

    def forward(self, data, meta_unrolls=1, num_theta_masks=1):
        # Extract relevant amounts
        ksp       = data['ksp']
        mask      = data['mask'] # 2D mask (or whatever, easy to adjust)

        # Initializers
        with torch.no_grad():
            # Initial maps
            if self.map_init == 'fixed':
                est_maps_kernel = self.init_maps_kernel
            elif self.map_init == 'estimated':
                # From dataloader
                est_maps_kernel = data['init_maps'].type(torch.complex64)
            elif self.map_init == 'espirit':
                # From dataloader
                est_maps_kernel = data['s_maps_cplx']

            # Initial image
            if self.img_init == 'fixed':
                est_img_kernel = sp.dirac(self.img_kernel_shape, dtype=np.complex64)[None, ...]
                est_img_kernel = np.repeat(est_img_kernel, self.batch_size, axis=0)
                # Image domain
                est_img_kernel = sp.ifft(est_img_kernel, axes=(-2, -1))
                est_img_kernel = torch.tensor(est_img_kernel, dtype=torch.cfloat).cuda()
            elif self.img_init == 'estimated':
                # Get adjoint map operator
                _, adjoint_ops, _ = \
                self.get_core_torch_ops(est_maps_kernel, None,
                                        mask, 'ConvSense') # Use all the masks to initialize
                adjoint_batch_op = self.get_batch_op(adjoint_ops, self.batch_size)
                # Apply
                est_img_kernel = adjoint_batch_op(ksp).type(torch.complex64)

        # Logging outputs
        if self.logging:
            # Kernels after denoiser modules
            img_kernel_denoised = []
            # Estimated logs
            img_logs, ksp_logs = [], []
            img_logs.append(copy.deepcopy(est_img_kernel))

        # !!! Only once is sufficient
        # Get operators for maps --> images using map kernel
        normal_ops, adjoint_ops, forward_ops = \
            self.get_core_torch_ops(est_maps_kernel, None,
                    mask, 'ConvSense')
        # Get joint batch operators for adjoint and normal
        normal_batch_op, adjoint_batch_op = \
            self.get_batch_op(normal_ops, self.batch_size), \
            self.get_batch_op(adjoint_ops, self.batch_size)

        # For each outer unroll
        for meta_idx in range(meta_unrolls):
            ## !!! Block 1
            if self.block1_max_iter > 0:
                assert False, 'Deprecated!'
                
            ## !!! Block 2
            # Compute RHS
            if meta_idx == 0:
                rhs = adjoint_batch_op(ksp) # flipped order of was ksp[:,mask_idx]: same below
            else:
                rhs = adjoint_batch_op(ksp) + \
                    self.block2_l2lam[0] * est_img_kernel

            # Get unrolled CG op
            cg_op = ZConjGrad(rhs, normal_batch_op,
                             l2lam=self.block2_l2lam[0],
                             max_iter=self.block2_max_iter,
                             eps=self.cg_eps, verbose=self.verbose)
            # Run CG
            est_img_kernel = cg_op(est_img_kernel)

            # Log
            if self.logging:
                img_logs.append(est_img_kernel)

            # Apply image denoising network in image space
            est_img_kernel = torch.view_as_real(est_img_kernel)
            est_img_kernel = self.image_net(est_img_kernel[None, ...])[0]
            est_img_kernel = torch.view_as_complex(est_img_kernel)
            
            # Log
            if self.logging:
                img_kernel_denoised.append(est_img_kernel)

                # For all unrolls, construct k-space
                if meta_idx < meta_unrolls - 1:
                    _, _, scratch_ops = \
                    self.get_core_torch_ops(est_maps_kernel, None,
                                            self.ones_mask, 'ConvSense')
                    scratch_batch_op = self.get_batch_op(scratch_ops, self.batch_size)
                    est_ksp = scratch_batch_op(est_img_kernel)
                    # Log
                    ksp_logs.append(est_ksp)

        # Compute output coils with an unmasked convolution operator
        _, _, forward_ops = \
                self.get_core_torch_ops(est_maps_kernel, None,
                                        self.ones_mask, 'ConvSense')
        forward_batch_op = self.get_batch_op(forward_ops, self.batch_size)
        est_ksp = forward_batch_op(est_img_kernel)

        if self.logging:
            # Add final ksp to logs
            ksp_logs.append(est_ksp)
            # Glue logs
            img_logs = torch.cat(img_logs, dim=0)
            img_kernel_denoised = torch.cat(img_kernel_denoised, dim=0)

        if self.logging:
            return est_img_kernel, est_maps_kernel, est_ksp, \
                img_logs, img_kernel_denoised, ksp_logs
        else:
            return est_img_kernel, est_maps_kernel, est_ksp
