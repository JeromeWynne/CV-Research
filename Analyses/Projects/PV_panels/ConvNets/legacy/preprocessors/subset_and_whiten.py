""" Default preprocessor.
    Subsets, whitens images.
"""

import numpy as np
from scipy.misc import imresize

##============= FUNCTIONS =================##
def extract_patches(images, masks, patch_size, ix):
    """
    	Extracts patches from image data. Retuns a stack of images.

    Args:
        images:     np.float32 array of images.
        masks:      np.float32 array of by-pixel class masks.
        patch_size: integer indicating kernel side length.
        ix:         Indices of pixels at which to sample.

    Returns:
        patches:    np.float32 of subset images.
        ohe_labels: np.float32 array of one-hot encoded by-pixel image labels
    """
    ps         = patch_size
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


def preprocessing(self, dataset, masks, mode, batch_size):
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
    # Resize images and masks passed
    MASK_THRESHOLD  = 0.2
    resized_dataset = np.array([imresize(img.squeeze(), [self.image_size, self.image_size]) for img in dataset])
    resized_masks   = np.array([imresize(msk, [self.image_size, self.image_size]) for msk in masks])
    resized_masks   = np.greater(resized_masks, MASK_THRESHOLD*255)
    resized_dataset = np.expand_dims(resized_dataset, axis = -1)

    # Sample a random selection of pixels
    if (mode == 'train') or (mode == 'valid'):
        ix = {'crack'   : np.array(np.nonzero(masks)).T,
              'nocrack' : np.array(np.nonzero(np.logical_not(masks))).T}

        ix['nocrack'] = ix['nocrack'][np.random.randint(low  = 0,
                                                        high = ix['nocrack'].shape[0],
                                                        size = batch_size//2),
                                      :]
        ix['crack']   = ix['crack'][np.random.randint(low  = 0,
                                                      high = ix['crack'].shape[0],
                                                      size = batch_size//2),
                                      :]
        ix = np.concatenate([ix['crack'], ix['nocrack']], axis = 0)
        ix = np.random.permutation(ix)

    else if (mode == 'test'):


    # Subtract mean and scale
    filtered_dataset = (subset_dataset.astype(np.float32)
                            - self.pp_parameters['mean'])/self.pp_parameters['std']
    filtered_labels  = subset_labels.astype(np.float32)

    return filtered_dataset, filtered_labels
