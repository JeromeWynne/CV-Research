"""pymv.classification"""
# Created:      11/07/2017
# Description:  Provides an interface to a class that can be used to
#               test machine vision classifiers.
# Last updated: 11/07/2017

import numpy as np

""" Module functions """
# > split_images

def split_images(dataset, labels, fraction=0.8):
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

    positive_ix   = np.array(np.nonzero(np.any(labels == 1., axis = 0))).T # Indices of images containing positive class
    negative_ix   = np.array(np.nonzero(np.any(labels == 0., axis = 0))).T # Indices of images containing negative class
    rnd_pos_ix    = np.random.choice(positive_ix,
                                     size    = int(fraction*positive_ix.shape[0]),
                                     replace = False) # Subset of posti indices
    rnd_neg_ix    = np.random.choice(negative_ix,
                                     size    = int(fraction*negative_ix.shape[0]),
                                     replace = False)
    mask          = np.zeros([labels.shape[0]]).astype(bool)
    mask[rnd_pos_ix] = True
    mask[rnd_neg_ix] = True
    train_images  = dataset[mask, :, :, :]
    train_labels  = dataset[mask, :]
    test_images   = dataset[np.logical_not(mask), :, :, :]
    test_labels   = dataset[np.logical_not(mask), :, :, :]

    return train_images, test_images, train_labels, test_labels



class Tester(object):
    """
    An agent for running and storing the results of tests.

    Methods:
        Holdout   : Evaluates the model using holdout validation.
        Bootstrap : Evaluates the model by measuring performance across bootstrap fits.

    Attributes:
        dataset       : np.float32 array (units x rows x cols x channels)
        labels        : one-hot encoded np.float32 array (units x rows x cols x channels)
        graph         : tensorflow graph (inc. processing of images)
        test_spec     : dictionary e.g. {'mode':'holdout', 'seed':1}
        test_results  : dictionary

     This library uses numpy and tensorflow for all operations.
    """

    def __init__(self, dataset, labels, graph, test_spec):
        self.dataset   = {'full':dataset}
        self.labels    = {'full':labels} # To begin with, we assume binary labelling
        self.graph     = graph
        self.test_spec = test_spec

        np.random.seed(self.test_spec['seed'])

        # Read the test spec. to determine the test to run.
        if self.test_spec['mode'] == 'holdout':   self.holdout()
        if self.test_spec['mode'] == 'bootstrap': self.bootstrap()


    def holdout(self):
        # Fits the model to a subset of the data then calculates its
        # performance on a held-out fraction of the data.

        # Subset the data
        ( self.dataset['train'], self.dataset['test'],
          self.labels['train'], self.labels['test']   ) = split_images(self.dataset['full'],
                                                                       self.labels['full'], fraction = 0.8)
        # Fit the model to the training data

    def bootstrap(self):
