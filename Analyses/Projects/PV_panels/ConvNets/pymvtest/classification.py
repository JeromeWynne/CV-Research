"""pymv.classification"""
# Created:      11/07/2017
# Description:  Provides an interface to a class that can be used to
#               test machine vision classifiers.
# Last updated: 11/07/2017

from scipy.misc import imresize
from os.path import exists
from os import makedirs
from shutil import rmtree
from sys import getsizeof
import tensorflow as tf
import numpy as np
import types
import json

""" Module functions """
# > split_dataset       - performs a stratified split of a dataset of images and label masks.
# > minibatch           - returns a subet of the data, parameterized by a step number.

""" Module classes """
# > Tester       - an agent for preprocessing data, running tests, and
#                  reporting test results.

def split_dataset(dataset, labels, fraction=0.8):
    """
    Splits the dataset images and labels according to the classes they contain.
    Args:
        dataset: np.float32 array of images                 [units x rows x cols x channels]
        labels:  np.float32 array of per-pixel label masks  [units x rows x cols x channels]
        fraction: float indicating training fractions       [scalar]
    Returns:
        Arrays containing the images and associated label masks.
            > train_images, test_images, train_labels, test_labels
        train_images: np.float32 array of image subset      [units x rows x cols x channels]
        test_images:  np.float32 array of image subset      [units x rows x cols x channels]
        train_labels: np.float32 array of per-pixel labels  [units x rows x cols x channels]
        test_labels:  np.float32 array of per-pixel labels  [units x rows x cols x channels]
    Raises:
        none
    """
    positive_ix   = np.array(np.nonzero(
                                np.sum(np.sum(labels == 1., axis = 2),
                                axis = 1))).flatten() # Indices of images containing positive class
    negative_ix   = np.array(np.nonzero(
                                np.sum(np.sum(labels == 0., axis = 2),
                                axis = 1))).flatten() # Indices of images containing negative class
    n = int(fraction*positive_ix.shape[0])
    rnd_pos_ix    = np.random.choice(positive_ix,
                                     size    = n,
                                     replace = False) # Subset of posti indices
    rnd_neg_ix    = np.random.choice(negative_ix,
                                     size    = n,
                                     replace = False)

    mask          = np.zeros([labels.shape[0]]).astype(bool)
    mask[rnd_pos_ix] = True
    mask[rnd_neg_ix] = True

    train_images  = dataset[mask, :, :, :]
    train_labels  = labels[mask, :, :]
    test_images   = dataset[np.logical_not(mask), :, :, :]
    test_labels   = labels[np.logical_not(mask), :, :]
    return train_images, test_images, train_labels, test_labels

def augment_images(data):
    if np.random.randint(2): data = np.flip(data, axis = 1)
    if np.random.randint(2): data = np.flip(data, axis = 2)
    return data

def resize(images, masks, image_size):
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
    MASK_THRESHOLD  = 0.1
    resized_images  = np.array([imresize(img.squeeze(), [image_size, image_size]) for img in images])
    resized_masks   = np.array([imresize(msk, [image_size, image_size]) for msk in masks])
    resized_masks   = np.greater(resized_masks, MASK_THRESHOLD)
    resized_images  = np.expand_dims(resized_images, axis = -1)
    return resized_images, resized_masks

def get_batch_indices(masks, batch_size):
    # Masks is a stack of full-sized binary masks.
    # Returns indices of pixels to sample at.]

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
    return ix

def get_patches(images, masks, ix, patch_size):
    # Accepts a stack of images, masks, and indices
    # Returns patches corresponding to each index
    ps         = patch_size
    patches    = np.zeros([ix.shape[0], patch_size, patch_size, 1])
    ohe_labels = np.zeros([ix.shape[0], 2])

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

def get_batch(images, masks, batch_size, patch_size):
    # Accepts a stack of images and binary masks.
    # Returns a stack of patches and a matrix of one-hot-encoded labels
    ix = get_batch_indices(masks, batch_size)
    batch_patches, batch_labels = get_patches(images, masks, ix, patch_size)
    return batch_patches, batch_labels

class Tester(object):
    """
    An agent for running and storing the results of tests on image classifiers.

    Methods:
        holdout   : Evaluates the model using holdout validation.
        bootstrap : Evaluates the model by measuring performance across bootstrap fits.

    Arguments:
        dataset       : np.float32 stack of images (NHWC)
        masks         : np.float32 stack of binary masks (NHW)
        TF            : Dictionary containing configuration and tensorflow variables
                        {
                         'seed':<int>,
                         'batch_size':<int>,
                         'training_steps':<int>,
                         'graph':<Python namespace for the TF graph>,
                         'optimizer':<Python namespace for optimizer operation>,
                         'loss':<Python namespace for loss tensor>,
                         'summary':<Python namespace for summary operation>,
                         'training_data':<Python namespace for training data tensor>,
                         'training_labels':<Python namespace for training labels tensor>,
                         'training_predictions':<Python namespace for training predictions tensor>,
                         'validation_data':<Python namespace for validation data tensor>
                         'validation_labels':<Python namespace for validation labels tensor<
                         'validation_predictions':<Python namespace for validation predictions tensor>,
                         }
        preprocessing : def func(self, dataset, masks, mode):
                        ...
                        return processed_images, ohe_labels

     This library uses numpy and tensorflow for all operations.
    """

    def __init__(self, dataset, masks, TF):
        self.dataset         = {'full':dataset} # An array of images
        self.labels          = {'full':masks}   # An array of per-pixel image labels
        self.seed            = TF['seed']       # Integer to initialize random number generator

        self.training_steps  = TF['training_steps'] # Integer number of training steps
        self.split_fraction  = TF['split_fraction'] # Train/validation split fraction (whole images)
        self.batch_size      = TF['batch_size']     # Integer batch size

        self.tf_loss          = TF['loss']           # Namespace of tensorflow loss tensor
        self.tf_graph         = TF['graph']          # A TensorFlow graph
        self.tf_optimizer     = TF['optimizer']      # Namespace of tensorflow optimizer operation
        self.tf_data          = TF['data']           # Namespace of tensorflow data
        self.tf_labels        = TF['labels']         # Namespace of tensorflow labels
        self.tf_predictions   = TF['predictions']    # Namespace of tensorflow predictions
        self.tf_accuracy      = TF['accuracy']       # Namespace of tensorflow accuracy
        self.tf_train_summary = TF['train_summary']
        self.tf_saver         = TF['saver']

        self.patch_size         = TF['patch_size']
        self.image_size         = TF['image_size'] # Determines what size to resize images to

        self.test_id            = TF['test_id']

        np.random.seed(self.seed)

        """
        Fits the model to a subset of the data, then calculates its
        performance on a held-out fraction of the data.
        """
        # Split the rest of the data by image class content
        ( self.dataset['train'], self.dataset['valid'],
          self.labels['train'],  self.labels['valid']  ) = split_dataset(self.dataset['full'],
                                                                         self.labels['full'],
                                                                         fraction = self.split_fraction)
        print('Training dataset dimensions: {}'.format(self.dataset['train'].shape))
        print('Validation dataset dimensions: {}'.format(self.dataset['valid'].shape))
        # NOTE: by-pixel label masks -> Preprocessor -> ohe label for each image returned

    def fit_model(self):
        """
        Instantiates and executes a tensorflow session with the Tester's graph.
        """
        with tf.Session(graph = self.tf_graph) as session:

            print('\nFitting model...')
            session.run(tf.global_variables_initializer())

            train_writer  = tf.summary.FileWriter('./results/'+self.test_id+'/train', session.graph)
            valid_writer  = tf.summary.FileWriter('./results/'+self.test_id+'/valid', session.graph)

            for step in range(self.training_steps):
                    if step % 100 == 0: # Validate
                        val_batch, val_labels = get_batch(self.dataset['valid'],
                                                          self.labels['valid'],
                                                          batch_size = 1000,
                                                          patch_size = self.patch_size)
                        val_fd  = { self.tf_data   : val_batch,
                                    self.tf_labels : val_labels }
                        s, val_acc, val_loss = session.run([self.tf_train_summary,
                                                            self.tf_accuracy, self.tf_loss],
                                                            feed_dict = val_fd)
                        if step != 0:
                            print('(Step {:^5d}) Minibatch accuracy: {:>8.2f}'.format(step, tr_acc))
                            print('(Step {:^5d}) Minibatch loss: {:>12.4f}'.format(step, l))
                        print('(Step {:^5d}) Validation accuracy: {:>7.2f}\n'.format(step, val_acc))
                        valid_writer.add_summary(s, step)

                    else: # Train
                        train_batch, train_labels = get_batch(self.dataset['train'],
                                                              self.labels['train'],
                                                              batch_size = self.batch_size,
                                                              patch_size = self.patch_size)
                        train_batch = augment_images(train_batch)
                        fd = { self.tf_data:train_batch,
                               self.tf_labels:train_labels }
                        s, _, l, tr_acc = session.run([self.tf_train_summary, self.tf_optimizer, self.tf_loss,
                                                       self.tf_accuracy], feed_dict = fd)
                        train_writer.add_summary(s, step)

            self.tf_saver.save(session, './checkpoints/'+self.test_id+'.ckpt')
            train_writer.close()
            valid_writer.close()

    def query_model(self, images, masks):
        # Accepts a stack of query images
        # Returns probability masks for each of those images
        with tf.Session(graph = self.tf_graph) as session:

            self.tf_saver.restore(session, './checkpoints/'+self.test_id+'.ckpt')

            predictions = np.zeros_like(images, dtype = np.float32).flatten()

            for j, ix in enumerate(np.array(np.nonzero(np.ones_like(masks))).T):
                ix               = np.array([ix])
                q_patch, q_label = get_patches(images, masks, ix, self.patch_size)
                fd               = {self.tf_data:q_patch, self.tf_labels:q_label}
                predictions[j]   = session.run(self.tf_predictions, feed_dict = fd)[0, 1]

            predictions = predictions.reshape(masks.shape)
        return predictions, masks

    def write_model_spec(self):
        # Write model spec. to text file
        with open('./results/'+self.test_id+'/'+self.test_id+'.txt', 'w') as file:
            test_spec = {
                         'test_id':self.test_id,
                         'model_id':self.model_id,
                         'preprocessor_id':self.preprocessor_id,
                         'seed':self.seed,
                         'split_fraction':self.split_fraction,
                         'batch_size':self.batch_size,
                         'patch_size':self.patch_size,
                         'image_size':self.image_size
                         }
            file.write(json.dumps(test_spec))
