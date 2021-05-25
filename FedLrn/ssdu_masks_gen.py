import numpy as np
import utils

class ssdu_masks():
    """

    Parameters
    ----------
    rho: split ratio for training and loss mask. \ rho = |\Lambda|/|\Omega|
    small_acs_block: keeps a small acs region fully-sampled for training masks
    if there is no acs region, the small acs block should be set to zero
    input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol

    Gaussian_selection:
    -divides acquired points into two disjoint sets based on Gaussian  distribution
    -Gaussian selection function has the parameter 'std_scale' for the standard deviation of the distribution. We recommend to keep it as 2<=std_scale<=4.

    Uniform_selection: divides acquired points into two disjoint sets based on uniform distribution

    Returns
    ----------
    trn_mask: used in data consistency units of the unrolled network
    loss_mask: used to define the loss in k-space

    """

    def __init__(self, rho=0.4, small_acs_block=(4, 4), num_theta=1, theta_fraction=1.):
        self.rho = rho
        self.small_acs_block = small_acs_block
        self.num_theta = num_theta
        assert 0 < theta_fraction <= 1, 'Incorrect sub-lambda fraction!'
        self.theta_fraction = theta_fraction

    def Gaussian_selection(self, input_mask, std_scale=4):
        # Only use mask
        nrow, ncol = input_mask.shape[-2], input_mask.shape[-1] #changed from 0 1 to match the h5py file notation we are using
        center_kx = int(np.ceil(nrow / 2.) - 1)
        center_ky = int(np.ceil(ncol / 2.) - 1)

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
                  center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0 #flipped kx and ky here

        loss_mask = np.zeros_like(input_mask)
        count = 0

        while count <= np.int(np.ceil(np.sum(input_mask[:]) * self.rho)):

            indy = np.int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale))) #flipped kx and ky here
            indx = np.int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))

            if (0 <= indy < nrow and 0 <= indx < ncol and temp_mask[indy, indx] == 1 and loss_mask[indy, indx] != 1):
                loss_mask[indy, indx] = 1
                count = count + 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask

    def uniform_selection(self, input_data, input_mask, num_iter=1):
        assert False, 'Deprecated'
        nrow, ncol = input_data.shape[0], input_data.shape[1]

        center_kx = int(utils.find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(utils.find_center_ind(input_data, axes=(0, 2)))

        if num_iter == 0:
            print(f'\n Uniformly random selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2: center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2: center_ky + self.small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                               size=np.int(np.count_nonzero(pr) * self.rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = utils.index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask
    
    # Draw i.i.d. subsets of 'trn_mask'
    def split_train_masks(self, trn_mask):
        # Only use mask
        nrow, ncol = trn_mask.shape[-2], trn_mask.shape[-1] #changed from 0 1 to match the h5py file notation we are using
        center_kx = int(np.ceil(nrow / 2.) - 1)
        center_ky = int(np.ceil(ncol / 2.) - 1)
        
        # All possible indexes
        candidate_idx = np.where(trn_mask.flatten() > 1e-10)[0]
        
        # Sample without replacement
        sampled_idx = [np.random.choice(
            candidate_idx, size=int(self.theta_fraction*len(candidate_idx)), 
            replace=False) for _ in range(self.num_theta)]
        
        # Safety check - distribute left out elements
        all_samples = np.unique(np.asarray(sampled_idx).flatten())
        left_out    = np.setdiff1d(candidate_idx, all_samples)
        
        if len(left_out) == 0:
            # Create a set of masks
            trn_masks = [trn_mask.flatten()[sampled_idx].reshape(trn_mask.shape)
                         for _ in range(self.num_theta)]
            
            return trn_masks
        else:
            # Shuffle and split evenly
            left_out     = left_out[np.random.permutation(len(left_out))]
            partitions   = np.asarray([int((idx+1) * len(left_out) / self.num_theta)
                                       for idx in range(self.num_theta-1)])
            splits       = np.split(np.arange(len(left_out)), partitions)
            # Add to indices
            sampled_idx  = [np.concatenate((sampled_idx[idx], left_out[splits[idx]]))
                            for idx in range(self.num_theta)]
            
            # Hard check that everything is sampled
            assert len(np.unique(np.hstack(sampled_idx).flatten())) == len(candidate_idx)
            
            # Create a set of masks
            trn_masks = []
            for mask_idx in range(self.num_theta):
                # Fill in pointwise multiplication pattern
                local_mask                        = np.zeros((len(trn_mask.flatten())))
                local_mask[sampled_idx[mask_idx]] = 1.
                
                # Apply multiplication and reshape
                lambda_mask = (trn_mask.flatten() * local_mask).reshape(trn_mask.shape)
                # Fill in ACS
                lambda_mask[center_kx - self.small_acs_block[0] // 2:
                                center_kx + self.small_acs_block[0] // 2,
                            center_ky - self.small_acs_block[1] // 2:
                                center_ky + self.small_acs_block[1] // 2] = 1.
                trn_masks.append(lambda_mask)
                
            # Hard check for leaking samples
            assert np.sum(sum(trn_masks) > 1e-10) == np.sum(trn_mask > 1e-10)
        
        return trn_masks

# Quick test
if False:
    num_theta = 6
    # Get a mask generator
    mask_gen = ssdu_masks(num_theta=num_theta, theta_fraction=0.5,
                          small_acs_block=(8, 8), rho=0.)
    # Reference mask
    full_mask = (np.random.uniform(size=(360)) > 0.5).astype(int)
    full_mask[360//2-13:360//2+13] = 1. # 26 ACS lines
    full_mask = np.asarray([full_mask] * 640)
    # Split Theta / Lambda
    train_mask, loss_mask = mask_gen.Gaussian_selection(full_mask)
    
    # Split Set-Theta
    theta_masks = mask_gen.split_train_masks(train_mask)
    
    # Plot
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(2, 4, 1); plt.imshow(train_mask, cmap='gray');
    plt.axis('off'); plt.title('Inner');
    
    plt.subplot(2, 4, 5); plt.imshow(loss_mask, cmap='gray');
    plt.axis('off'); plt.title('Outer');
    
    for mask_idx in range(num_theta):
        plot_idx = mask_idx + 2 if mask_idx < 3 else mask_idx + 3
        plt.subplot(2, 4, plot_idx); plt.imshow(
            theta_masks[mask_idx], cmap='gray');
        plt.axis('off'); plt.title('Inner %d' % (mask_idx+1))
        
    plt.savefig('mask_debugging.png', dpi=300)
    plt.close()