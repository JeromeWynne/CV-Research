""" Default preprocessor.
    Subsets, whitens images.
"""

import numpy as np
from scipy.misc import imresize

##============= FUNCTIONS =================##
def extract_patches(images, masks, patch_size, mode, n_samples=None, query_side = None):
    """
    	Extracts patches from image data.
	If training, returns a balanced set of patches.
    	If testing, returns patches for every pixel in every image passed.

    Args:
        images:     np.float32 array of images.
        masks:      np.float32 array of by-pixel class masks.
        patch_size: integer indicating kernel side length.
        train:      bool indicating whether to return a balanced dataset.
        n_samples:  number of samples to draw per class

    Returns:
        patches:    np.float32 of subset images.
        ohe_labels: np.float32 array of one-hot encoded by-pixel image labels
    """
    ps = patch_size
    # 1. Get indices of pixels to sample
    # Training/validation mode balances dataset
    if (mode == 'train') or (mode == 'valid'):

        ix = {'crack' : np.array(np.nonzero(masks)).T,
              'nocrack' : np.array(np.nonzero(np.logical_not(masks))).T}

        if n_samples == None: n_samples = ix['crack'].shape[0]

        ix['nocrack'] = ix['nocrack'][np.random.randint(low  = 0,
                                                        high = ix['nocrack'].shape[0],
                                                        size = n_samples),
                                      :]
        ix['crack']   = ix['crack'][np.random.randint(low  = 0,
                                                      high = ix['crack'].shape[0],
                                                      size = n_samples),
                                      :]
        ix = np.concatenate([ix['crack'], ix['nocrack']], axis = 0)
        ix = np.random.permutation(ix)
    # Testing mode does not balance dataset (i.e. extracts patches centered on all pixels)
    elif mode == 'test':
            crop_region = np.zeros_like(masks)
            crop_region[:, :query_side, :query_side] = 1. # TF['query_side']
            ix = np.array(np.nonzero(crop_region)).T

    patches    = np.zeros([ix.shape[0], patch_size, patch_size, 1])
    ohe_labels = np.zeros([ix.shape[0], 2])

    # 2. Sample patches centered on selected pixels
    for j, i in enumerate(ix):

        if (j == 0) or np.any(images[i[0], :, :, :] != img):
            img  = images[i[0], :, :, :]
            pimg = np.pad(img, pad_width = [[ps, ps], [ps, ps], [0, 0]],
                          mode = 'constant', constant_values = 0)

        patches[j, :, :, :] =  pimg[i[1] + ps // 2 : i[1] + (3 * ps // 2),
                                    i[2] + ps // 2 : i[2] + (3 * ps // 2)]
        k = masks[i[0], i[1], i[2]]
        ohe_labels[j, int(k)] = 1. # One-hot encode the masks

    return patches, ohe_labels


def preprocessing(self, dataset, masks, mode, n_samples=None):
    """
    Extracts and whitens patches. Designed in accordance with Tester requirements.

    Arguments:
        dataset: np.float32 array of images.
        labels:  np.float32 array of by-pixel masks for the images.
        train:   bool indicating whether to balance dataset and store filter parameters.
        n_sampleS: # samples per class ('train' and 'valid' modes only)

    Returns:
        filtered_dataset: subset images.
        filtered_labels:  one-hot encoded labels
    """
    # Resizing
    MASK_THRESHOLD  = 0.2
    resized_dataset = np.array([imresize(img.squeeze(), [self.image_size, self.image_size]) for img in dataset])
    resized_masks   = np.array([imresize(msk, [self.image_size, self.image_size]) for msk in masks])
    resized_masks   = np.greater(resized_masks, MASK_THRESHOLD*255)

    resized_dataset = np.expand_dims(resized_dataset, axis = -1)

    # Subsetting

    # Filtering
    if mode == 'train':
        subset_dataset, subset_labels  = extract_patches(resized_dataset,
                                                         resized_masks, self.patch_size,
                                                         mode = 'train', n_samples = n_samples) # Balanced subset
        self.pp_parameters['mean'] = np.mean(subset_dataset, axis = 0)
        self.pp_parameters['std']  = np.std(subset_dataset, axis = 0)

    elif mode == 'valid':
        subset_dataset, subset_labels  = extract_patches(resized_dataset,
                                                         resized_masks, self.patch_size,
                                                         mode = 'valid', n_samples = n_samples)
    elif mode == 'test':
        subset_dataset, subset_labels  = extract_patches(resized_dataset,
                                                         resized_masks, self.patch_size,
                                                         mode = 'test')
    # Subtract mean and scale
    filtered_dataset = (subset_dataset.astype(np.float32)
                            - self.pp_parameters['mean'])/self.pp_parameters['std']
    filtered_labels  = subset_labels.astype(np.float32)

    return filtered_dataset, filtered_labels
