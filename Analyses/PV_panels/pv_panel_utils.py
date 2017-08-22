from glob import glob
import numpy as np
from scipy.misc import imread

def _get_name(filepath, mode):
    if mode == 'ubuntu':
        return filepath.split('/')[-1]
    if mode == 'windows':
        return filepath.split('\\')[-1]

def import_data(mode = 'ubuntu'):
    n_cracked   = len(glob('./Data/170729_Panel_Solos_Resized/CrackedScores/*.png'))
    n_query     = 3

    cracked_score_fps   = glob('./Data/170729_Panel_Solos_Resized/CrackedScores/*.png')
    uncracked_score_fps = glob('./Data/170729_Panel_Solos_Resized/UncrackedScores/*.png') 

    image_filepaths     = (['./Data/170729_Panel_Solos_Resized/Cracked/' + _get_name(fp, mode) 
                             for fp in cracked_score_fps] +
                           ['./Data/170729_Panel_Solos_Resized/Uncracked/' + _get_name(fp, mode)
                             for fp in uncracked_score_fps])

    score_filepaths  = ( cracked_score_fps +
                          uncracked_score_fps )

    scores  = np.expand_dims(np.array(
                    [imread(fp) for fp in score_filepaths], dtype = 'float32'), axis = -1) / 255
    scores  = np.concatenate([scores, 1 - scores], axis = -1) # 'One-hot' encode scores
    images  = np.expand_dims(np.array(
                    [imread(fp) for fp in image_filepaths], dtype = 'float32'), axis = -1)
    images  = (images - 128)/128

    query_ix     = np.random.randint(0, n_cracked, 3)
    query_mask   = np.zeros([scores.shape[0]], dtype = bool)
    query_mask[query_ix] = True

    query_scores = scores[query_mask, :, :, :]
    query_images = images[query_mask, :, :, :]
    scores       = scores[np.logical_not(query_mask), :, :, :]
    images       = images[np.logical_not(query_mask), :, :, :]

    labels       = (np.sum(scores, axis = (1, 2)) > 0).squeeze()[:, 0]
    
    return images, scores, labels, query_images, query_scores