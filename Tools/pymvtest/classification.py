"""pymv.classification"""
# Created:      11/07/2017
# Description:  Provides an interface to a class that can be used to
#               test machine vision classifiers.
# Last updated: 11/07/2017

from sys import getsizeof
import tensorflow as tf
import numpy as np
import types

""" Module functions """
# > split_dataset       - performs a stratified split of a dataset of images and label masks.
# > minibatch           - returns a subet of the data, parameterized by a step number.
# > accuracy_score      - returns the fraction of correct predictions made by the classifier.
# > ohe_mask            - one-hot encode a set of image masks by pixel.
# > whiten              - subtract mean and standardize each feature in a stack of images.
# > balance_dataset     - balance dataset classes by balanced resampling by one-hot encoded labels.

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

    positive_ix   = np.array(np.nonzero(np.sum(np.sum(labels == 1., axis = 2), axis = 1))).flatten() # Indices of images containing positive class
    negative_ix   = np.array(np.nonzero(np.sum(np.sum(labels == 0., axis = 2), axis = 1))).flatten() # Indices of images containing negative class

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
    train_labels  = labels[mask, :, :]
    test_images   = dataset[np.logical_not(mask), :, :, :]
    test_labels   = labels[np.logical_not(mask), :, :]

    return train_images, test_images, train_labels, test_labels

def minibatch(dataset, labels, batch_size, step):
    """
    Returns a subset of the dataset according to the training step.
    Args:
        dataset:    np.float32 array of images                         [units x rows x cols x channels]
        labels:     np.float32 array of one-hot-enocoded image labels  [units x classes]
        batch_size: int number of elements retrieved per step          [integer]
        step:       int step number in training session                [integer]
    Returns:
        A subset of the images and their associated labels.
            > batch_data, batch_labels
    Raises:
        none
    """
    posn         = (step * batch_size) % (dataset.shape[0] - batch_size)
    batch_data   = dataset[posn:(posn + batch_size), :, :, :]
    batch_labels = labels[posn:(posn + batch_size), :]
    return batch_data, batch_labels

def accuracy_score(predictions, ground_truth):
    """
    Accepts one-hot encoded predictions and ground truth. (np.float32 arrays)
    """
    score = 100*np.sum(np.equal(np.argmax(predictions, axis = 1),
                                np.argmax(ground_truth, axis = 1)))/ground_truth.shape[0]
    return score

def ohe_mask(mask):
    """
        Returns an array of one-hot encoded labels for each pixel in a stack of masks.
        Args:
            mask: np.float32 array of class masks (NHW)
        Returns:
            ohe_labels: np.float32 array of one-hot encoded labels (N*H*W, n_classes)
    """
    n_classes  = np.unique(mask).shape[0]
    ohe_labels = np.zeros([mask.shape[0]*mask.shape[1]*mask.shape[2], n_classes])
    for k in range(n_classes):
        ohe_labels[np.equal(mask, k).flatten(), k] = k
    return ohe_labels.astype(np.float32)

def balance_dataset(dataset, ohe_labels, n_samples):
    """
        Aggressively balances a dataset of classified images *with replacement*.
        Arguments:
            dataset:    np.float32 array of images (N_1 x H x W x C)
            ohe_labels: np.float32 array of one-hot encoded image labels. (N_1 x n_classes)
        Returns:
            balanced_dataset: np.float32 array of images (N_2 x H x W x C)
            balanced_labels:  np.float32 array of labels (N_2 x n_classes)
    """
    indices      = np.nonzero(dataset[:, 0, 0, 0] != None)
    index_probabilities = np.sum(ohe_labels / np.sum(ohe_labels, axis = 0),
                                 axis = 1)
    index_probabilities = index_probabilities / np.sum(index_probabilities)
    chosen_indices      = np.random.choice(indices, size = n_samples, replace = True,
                                           p = index_probabilities)

    balanced_dataset    = dataset[chosen_indices, :, :, :]
    balanced_labels     = ohe_labels[chosen_indices, :]
    return balanced_dataset, balaced_labels


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
                        {'mode':<string 'holdout'>,
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

    def __init__(self, dataset, masks, TF, preprocessor):
        self.dataset         = {'full':dataset} # An array of images
        self.labels          = {'full':masks}   # An array of per-pixel image labels
        self.mode            = TF['mode']       # String in {'holdout', 'bootstrap'}
        self.seed            = TF['seed']       # Integer to initialize random number generator
        self.preprocessor    = types.MethodType(preprocessor, self)    # A preprocessing function - see the example
        self.pp_parameters   = {}                   # Populated by tne preprocessor function (..., store_params = True)
        self.training_steps  = TF['training_steps'] # Integer number of training steps
        self.split_fraction  = TF['split_fraction'] # Train/validation split fraction (whole images)
        self.batch_size      = TF['batch_size']     # Integer batch size
        self.tf_loss         = TF['loss']           # Namespace of tensorflow loss tensor
        self.tf_summary      = TF['summary']        # Namespace of merged summary variables
        self.tf_graph        = TF['graph']          # A TensorFlow graph
        self.tf_optimizer    = TF['optimizer']      # Namespace of tensorflow optimizer operation
        self.tf_data         = TF['data']        # Namespace of tensorflow data
        self.tf_labels       = TF['labels']      # Namespace of tensorflow labels
        self.tf_predictions  = TF['predictions'] # Namespace of tensorflow predictions
        self.tf_accuracy     = TF['accuracy']    # Namespace of tensorflow accuracy

        np.random.seed(self.seed)

        # Read the test spec. to determine the test to run.
        if self.mode == 'holdout':
            self.holdout()


    def holdout(self):
        """
        Fits the model to a subset of the data, then calculates its
        performance on a held-out fraction of the data.
        """
        # Split the data by image class content
        ( self.dataset['train'], self.dataset['valid'],
          self.labels['train'],  self.labels['valid']  ) = split_dataset(self.dataset['full'],
                                                                         self.labels['full'],
                                                                         fraction = self.split_fraction)

        # Apply the filter and subsetting - configure the preprocessor on the training data.
        self.dataset['ptrain'], self.labels['ohetrain'] = self.preprocessor(self.dataset['train'],
                                                                self.labels['train'], mode = 'train')
        self.dataset['pvalid'], self.labels['ohevalid']  = self.preprocessor(self.dataset['valid'],
                                                                self.labels['valid'], mode = 'valid')
        # NOTE: by-pixel label masks -> Preprocessor -> ohe label for each image returned


    def evaluate_model(self):
        """
        Instantiates and executes a tensorflow session with the Tester's graph.
        """
        with tf.Session(graph = self.tf_graph) as session:

            print('\nFitting model...')
            session.run(tf.global_variables_initializer())
            train_writer  = tf.summary.FileWriter('FileWriterOutput/train', session.graph)
            valid_writer  = tf.summary.FileWriter('FileWriterOutput/valid', session.graph)

            for step in range(self.training_steps):
                    if step % 100 == 0:
                        val_fd  = { self.tf_data : self.dataset['pvalid'],
                                    self.tf_labels : self.labels['ohevalid'] }
                        s, val_acc = session.run([self.tf_summary, self.tf_accuracy],
                                                  feed_dict = val_fd)
                        if step != 0:
                            print('(Step {:^5d}) Minibatch accuracy: {:>7.2f}%'.format(step, tr_acc))
                            print('(Step {:^5d}) Minibatch loss: {:>12.4f}'.format(step, l))

                        print('(Step {:^5d}) Validation accuracy: {:>6.2f}%\n'.format(step, val_acc))
                        
                        valid_writer.add_summary(s, step)
                    else:
                        batch_data, batch_labels = minibatch(self.dataset['ptrain'], self.labels['ohetrain'],
                                                             self.batch_size, step)
                        fd = {self.tf_data:batch_data, self.tf_labels:batch_labels}
                        s, _, l, tr_acc = session.run([self.tf_summary, self.tf_optimizer, self.tf_loss,
                                                       self.tf_accuracy], feed_dict = fd)
                        train_writer.add_summary(s, step)

            writer.close()
