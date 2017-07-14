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
    Attributes:
        dataset       : np.float32 array (units x rows x cols x channels)
        labels        : one-hot encoded np.float32 array (units x rows x cols)
        graph         : tensorflow graph (inc. processing of images)
        spec          : dictionary e.g. {'mode':'holdout', 'seed':1,
                                         'n_train':1000, 'batch_size':32,
                                         'training_steps':10000}
        results       : dictionary containing test results
     This library uses numpy and tensorflow for all operations.
    """

    def __init__(self, dataset, masks, graph, optimizer, loss, spec, preprocessor):
        self.dataset         = {'full':dataset} # An array of images
        self.labels          = {'full':masks}  # An array of per-pixel image labels
        self.graph           = graph            # A TensorFlow graph
        self.optimizer       = optimizer
        self.loss            = loss
        self.preprocessor    = types.MethodType(preprocessor, self)    # A configured Preprocessor object - defaults
        self.spec            = spec             # A dictionary specifying the test config.
        self.pp_parameters   = {}               # Populated by preprocessor(..., store_params = True)

        np.random.seed(self.spec['seed'])

        # Read the test spec. to determine the test to run.
        if self.spec['mode'] == 'holdout':
            self.holdout()

    def holdout(self):
        # Fits the model to a subset of the data then calculates its
        # performance on a held-out fraction of the data.

        # Split the data by image class content
        ( self.dataset['train'], self.dataset['valid'],
          self.labels['train'],  self.labels['valid']  ) = split_dataset(self.dataset['full'],
                                                                        self.labels['full'], fraction = 0.8)

        # Apply the filter and subsetting - configure the preprocessor on the training data.
        self.dataset['ptrain'], self.labels['ohetrain'] = self.preprocessor(self.dataset['train'],
                                                                self.labels['train'], mode = 'train')
        self.dataset['pvalid'], self.labels['ohevalid']  = self.preprocessor(self.dataset['valid'],
                                                                self.labels['valid'], mode = 'valid')

        # NOTE: by-pixel label masks -> Preprocessor -> ohe label for each image returned
        # NOTE: Preprocessed data is stored to avoid low-level (i.e pixel neighborhood)
        #       class imbalance problems during training. Test data is left imbalanced.
        # NOTE: The number of training instances to be used should be used by preprocessing()
        #       and should be specified in spec (e.g. spec = { ... 'ntrain':5000})

    def evaluate_model(self):
        with tf.Session(graph = self.graph) as session:
            # Initialize model variables
            print('\nFitting model...')
            tf.global_variables_initializer().run()

            # Fit the model
            for step in range(self.spec['training_steps']):
            # NOTE: WHY IS THE VALIDATION ACCURACY SO BAD?????
                    batch_data, batch_labels = minibatch(self.dataset['ptrain'], self.labels['ohetrain'],
                                                         self.spec['batch_size'], step)
                    fd = {'tf_train_data:0':batch_data, 'tf_train_labels:0':batch_labels}
                    _, l, pred = session.run([self.optimizer,
                                              self.loss,
                                              'tf_train_predictions:0'], feed_dict = fd)

                    if step % 500 == 0:
                        print('Step {}\n-------------'.format(step))
                        print('Minibatch accuracy: {:04.2f}%'.format(accuracy_score(pred, batch_labels)))
                        print('Minibatch loss: {:06.4f}'.format(l))
                        validation_predictions = session.run('tf_test_predictions:0',
                                                             feed_dict = {'tf_test_data:0':self.dataset['pvalid']})
                        print('Validation accuracy: {:04.2f}%\n'.format(accuracy_score(
                                                             validation_predictions, self.labels['ohevalid'])))
