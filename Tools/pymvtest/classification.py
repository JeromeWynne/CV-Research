"""pymv.classification"""
# Created:      11/07/2017
# Description:  Provides an interface to a class that can be used to
#               test machine vision classifiers.
# Last updated: 11/07/2017

import tensorflow as tf
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

class Preprocessor(object):
    """
    An agent for processing images and storing image metrics.

    Atrributes:
        filter_parameters : dictionary containing a dataset's parameters
                             these can be assigned and used by a Preprocessor's functions.
    Methods:
        filter_and_subset  : Applies filters to the image, then subsets the resulting images.

    filter_config is a function that returns:
        A filter to be applied to each image in the dataset.
        This filter should accept a dataset and a boolean indicating
        whether to create a dictionary of config parameters.
        A set of parameters to be used in these filters

    def filter_function(self, dataset, use_existing_config):
        if not use_existig_config:
            self.filter_parameters['mean'] = np.mean(dataset, axis = 1)
            self.filter_parameters['std']  = np.std(dataset, axis = 1)
        filtered_dataset = (dataset - self.filter_parameters['mean'])/self.filter_parameters['std']
        return filtered_dataset

    def subset_function(self, dataset, label_masks):
        return dataset, label_masks

    """

    def __init__(self, filter_function, subset_function):
        self.filter = types.MethodType(filter_function, self) # A function accepting a stack of images and returning another stack of images
        self.subset = types.MethodType(subset_funciton, self) # A function accepting a stack of images and returning another stack of images
        self.filter_parameters = {}

    def apply_filter(self, dataset, use_existing_config = False):
        filtered_dataset = self.filter(dataset, use_existing_config)
        return filtered_dataset

    def apply_subset(self, dataset, label_masks):
        subset_dataset, subset_labels = self.subset(dataset, label_masks)
        return subset_dataset, subset_labels

class Tester(object):
    """
    An agent for running and storing the results of tests on image classifiers.

    Methods:
        Holdout   : Evaluates the model using holdout validation.
        Bootstrap : Evaluates the model by measuring performance across bootstrap fits.

    Attributes:
        dataset       : np.float32 array (units x rows x cols x channels)
        labels        : one-hot encoded np.float32 array (units x rows x cols)
        graph         : tensorflow graph (inc. processing of images)
        spec          : dictionary e.g. {'mode':'holdout', 'seed':1}
        results       : dictionary containing test results

     This library uses numpy and tensorflow for all operations.
    """

    def __init__(self, dataset, labels, graph, preprocess, spec):
        self.dataset         = {'full':dataset} # An array of images
        self.labels          = {'full':labels}  # An array of per-pixel image labels
        self.graph           = graph            # A TensorFlow graph
        self.preprocessor    = preprocessor     # A Preprocessor object
        self.spec            = spec             # A dictionary specifying the test config.

        np.random.seed(self.spec['seed'])

        # Read the test spec. to determine the test to run.
        if self.spec['mode'] == 'holdout':   self.holdout()
        if self.spec['mode'] == 'bootstrap': self.bootstrap()


    def holdout(self):
        # Fits the model to a subset of the data then calculates its
        # performance on a held-out fraction of the data.

        # Subset the data by image class content
        ( self.dataset['train'], self.dataset['test'],
          self.labels['train'], self.labels['test']   ) = split_images(self.dataset['full'],
                                                                       self.labels['full'], fraction = 0.8)

        self.dataset['ptrain'] = self.preprocessor.filter(self.dataset['train'], store_params = True) # Applies filters to whole images, then retrieves a balanced subset of the results
        self.dataset['ptest']  = self.preprocessor.

        # Test the model
        self.results = self.evaluate_model()

    #def bootstrap(self):


    def evaluate_model(self):
        with tf.Session(graph = self.graph) as session:
            print('Testing session initialized. Fitting model...')
            tf.global_variables_initializer().run()

            # Fit the model
            for step in range(spec['training_steps']):
                    batch_data, batch_labels = minibatch(self.dataset['train'], self.labels['train'],
                                                            self.spec['batch_size'], step)
                    batch_data = self.preprocess(batch_data)
                    fd = {tf_train_data:batch_data, tf_train_labels:batch_labels}

                    _  = session.run([optimizer, loss, tf_train_predictions], feed_dict = fd)

            # Evaluate the model's testing performance
            test_predictions = tf_test_predictions.eval(feed_dct = {tf_test_data:self.preprocess(self.dataset['test'])})
            test_accuracy    = np.sum(np.argmax(test_predictions, axis = 1) ==
                                        np.argmax(labels['test'], axis = 1))/labels['test'].shape[0]
            test_predictions =

        return {'predictions' : test_predictions, 'accuracy':test_accuracy}
