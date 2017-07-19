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

def split_dataset(dataset, labels, fraction=0.8, n=None):
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


    if n is not None:
            test_ix  = np.random.randint(0, dataset.shape[0], n)
            train_ix = [i for i in range(dataset.shape[0]) if i not in test_ix]
            train_images = dataset[train_ix, :, :, :]
            train_labels = labels[train_ix, :, :]
            test_images  = dataset[test_ix, :, :, :]
            test_labels  = labels[test_ix, :, :]

    if n is None:
            positive_ix   = np.array(np.nonzero(
                                        np.sum(np.sum(labels == 1., axis = 2), axis = 1))).flatten() # Indices of images containing positive class
            negative_ix   = np.array(np.nonzero(
                                        np.sum(np.sum(labels == 0., axis = 2), axis = 1))).flatten() # Indices of images containing negative class
            size = int(fraction*positive_ix.shape[0])
            rnd_pos_ix    = np.random.choice(positive_ix,
                                             size    = size,
                                             replace = False) # Subset of posti indices
            rnd_neg_ix    = np.random.choice(negative_ix,
                                             size    = size,
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

def augment_images(dataset):
    """
        Augments images. Configure as needed. Dataset is a stack of images.
    """
    if np.random.randint(2): dataset = np.flip(dataset, axis = 1)
    if np.random.randint(2): dataset = np.flip(dataset, axis = 2)
    return dataset


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
        self.tf_graph        = TF['graph']          # A TensorFlow graph
        self.tf_optimizer    = TF['optimizer']      # Namespace of tensorflow optimizer operation
        self.tf_data         = TF['data']        # Namespace of tensorflow data
        self.tf_labels       = TF['labels']      # Namespace of tensorflow labels
        self.tf_predictions  = TF['predictions'] # Namespace of tensorflow predictions
        self.tf_accuracy     = TF['accuracy']    # Namespace of tensorflow accuracy
        self.tf_train_summary = TF['train_summary']
        self.tf_test_summary  = TF['test_summary']

        np.random.seed(self.seed)

        # Read the test spec. to determine the test to run.
        if self.mode == 'holdout':
            self.holdout()


    def holdout(self):
        """
        Fits the model to a subset of the data, then calculates its
        performance on a held-out fraction of the data.
        """
        # Extract one query image for testing
        ( self.dataset['full'], self.dataset['test'],
          self.labels['full'], self.labels['test']    ) = split_dataset(self.dataset['full'],
                                                                        self.labels['full'],
                                                                        n = 1)

        # Split the rest of the data by image class content
        ( self.dataset['train'], self.dataset['valid'],
          self.labels['train'],  self.labels['valid']  ) = split_dataset(self.dataset['full'],
                                                                         self.labels['full'],
                                                                         fraction = self.split_fraction)

        # Apply the filter and subsetting - configure the preprocessor on the training data.
        self.dataset['ptrain'], self.labels['ohetrain'] = self.preprocessor(self.dataset['train'],
                                                                self.labels['train'], mode = 'train')
        self.dataset['pvalid'], self.labels['ohevalid']  = self.preprocessor(self.dataset['valid'],
                                                                self.labels['valid'], mode = 'valid')
        self.dataset['ptest'], self.labels['ohetest']    = self.preprocessor(self.dataset['test'],
                                                                self.labels['test'],  mode = 'test')
        # NOTE: by-pixel label masks -> Preprocessor -> ohe label for each image returned

        # Write preprocessed image locally
        from scipy.misc import imsave
        test_img = (self.dataset['ptest']*self.pp_parameters['std'] + self.pp_parameters['mean'])[:, 10, 10, 0]
        training_img = (self.dataset['ptrain']*self.pp_parameters['std'] + self.pp_parameters['mean'])[10, :, :, 0]
        imsave('./train_img.png', training_img)
        print(test_img)
        imsave('./test_img.png', np.reshape(test_img, [100, 100]))


    def evaluate_model(self):
        """
        Instantiates and executes a tensorflow session with the Tester's graph.
        """
        with tf.Session(graph = self.tf_graph) as session:

            print('\nFitting model...')
            session.run(tf.global_variables_initializer())
            train_writer  = tf.summary.FileWriter('FileWriterOutput/train', session.graph)
            valid_writer  = tf.summary.FileWriter('FileWriterOutput/valid', session.graph)
            test_writer   = tf.summary.FileWriter('FileWriterOutput/test',  session.graph)

            for step in range(self.training_steps):
                    if step % 100 == 0:
                        val_fd  = { self.tf_data : self.dataset['pvalid'],
                                    self.tf_labels : self.labels['ohevalid'] }
                        s, val_acc, val_loss = session.run([self.tf_train_summary,
                                                            self.tf_accuracy, self.tf_loss],
                                                            feed_dict = val_fd)
                        if step != 0:
                            print('(Step {:^5d}) Minibatch accuracy: {:>8.2f}'.format(step, tr_acc))
                            print('(Step {:^5d}) Minibatch loss: {:>12.4f}'.format(step, l))

                        print('(Step {:^5d}) Validation accuracy: {:>7.2f}\n'.format(step, val_acc))

                        valid_writer.add_summary(s, step)

                    else:
                        batch_data, batch_labels = minibatch(self.dataset['ptrain'], self.labels['ohetrain'],
                                                             self.batch_size, step)
                        batch_data = augment_images(batch_data)
                        fd = {self.tf_data:batch_data, self.tf_labels:batch_labels}
                        s, _, l, tr_acc = session.run([self.tf_train_summary, self.tf_optimizer, self.tf_loss,
                                                       self.tf_accuracy], feed_dict = fd)
                        train_writer.add_summary(s, step)

            # Testing
            test_fd = { self.tf_data : self.dataset['ptest'],
                         self.tf_labels : self.labels['ohetest']}
            s       = session.run(self.tf_test_summary, feed_dict = test_fd)
            test_writer.add_summary(s, self.training_steps)

            train_writer.close()
            valid_writer.close()
            test_writer.close()
