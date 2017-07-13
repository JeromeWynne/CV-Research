"""pymv.classification"""
# Created:      11/07/2017
# Description:  Provides an interface to a class that can be used to
#               test machine vision classifiers.
# Last updated: 11/07/2017

from sys import getsizeof
import tensorflow as tf
import numpy as np

""" Module functions """
# > split_dataset       - performs a stratified split of a dataset of images and label masks.
# > minibatch           - returns a subet of the data, parameterized by a step number.

""" Module classes """
# > Preprocessor - an agent for filtering and subsetting images.
#                  Allows filter parameters to be stored. A Preprocessor is
#                  used to initialize a Tester instance.
# > Tester       - an agent for running and storing the results of tests.


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

def accuracy_score(predictions, ground_truth):
    """
    Accepts one-hot encoded predictions and ground truth. (np.float32 arrays)
    """
    score = np.sum(np.argmax(test_predictions, axis = 1) ==
                            np.argmax(labels['test'], axis = 1))/labels['test'].shape[0]
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

def whiten(self, dataset, labels, train):
    """
    Default Preprocessor prepocessing (i.e. filter + subset) train. "self" refers to the
    Preprocessor instance that calls this function. If train == True,
    then the function configures the Preprocessor instance's pp_parameters{} attribute.
    This is so that we can configure a preprocessor with a training dataset, then use that
    configuration with a testing dataset.

    Args:
        dataset:    np.float32 array of images (NHWC)
        labels:     np.float32 array of pixel class masks  (NHWC)
        train:      bool indicating whether to configure Preprocessor.pp_parameters[]
                    and to return a balanced subset of data.

    Returns:
        filtered_dataset: np.float32 array of filtered and subset images.
        filtered_labels:  np.float32 array of one-hot-encoded labels for the subset images.
    """
    if train:
        self.pp_parameters['mean'] = np.mean(dataset, axis = 1)
        self.pp_parameters['std']  = np.std(dataset, axis = 1)

    filtered_dataset = (dataset - self.filter_parameters['mean'])/self.filter_parameters['std']
    filtered_labels  = ohe_mask(labels[:, 0, 0]) # One label per each image returned

    if train:
        filtered_dataset, filtered_labels = balance_dataset(filtered_dataset,
                                                                filtered_labels, self.spec['n_train'])

    return filtered_dataset, filtered_labels

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
    index_probabilities = np.sum(ohe_labels / np.sum(ohe_labels, axis = 0)),
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

    def __init__(self, dataset, labels, graph, preprocessor = whiten, spec):
        self.dataset         = {'full':dataset} # An array of images
        self.labels          = {'full':labels}  # An array of per-pixel image labels
        self.graph           = graph            # A TensorFlow graph
        self.preprocessor    = preprocessor     # A configured Preprocessor object - defaults
        self.spec            = spec             # A dictionary specifying the test config.
        self.pp_parameters   = {}               # Populated by preprocessor(..., store_params = True)

        np.random.seed(self.spec['seed'])

        # Read the test spec. to determine the test to run.
        if self.spec['mode'] == 'holdout':
            print('Holdout testing mode initialized.')
            self.holdout()
        if self.spec['mode'] == 'bootstrap':
            print('Bootstrap testing mode initialized.')
            self.bootstrap()

    def holdout(self):
        # Fits the model to a subset of the data then calculates its
        # performance on a held-out fraction of the data.

        print('\nSplitting images by class content...')
        # Split the data by image class content
        ( self.dataset['train'], self.dataset['test'],
          self.labels['train'],  self.labels['test']  ) = split_dataset(self.dataset['full'],
                                                                       self.labels['full'], fraction = 0.8)
        print('\n\nDataset \t Dim. \t Mem. Usage \n')
        print(' Train  \t {} \t {:06.2f}\n'.format(self.dataset['train'].shape),
                                                   getsizeof(self.dataset['train'])/(10**(-6)))
        print(' Test   \t {} \t {:06.2f}\n'.format(self.dataset['test'].shape),
                                                   getsizeof(self.dataset['test']/(10**(-6)))

        # Apply the filter and subsetting - configure the preprocessor on the training data.
        print('\nApplying preprocessing to training data.')
        self.dataset['ptrain'], self.labels['ohetrain'] = self.preprocessor(self.dataset['train'],
                                                                self.labels['train'], train = True)
        print('\nApplying preprocessing to testing data.')
        self.dataset['ptest'], self.labels['ohetest']  = self.preprocessor(self.dataset['test'],
                                                                self.labels['test'], train = False)
        print('\n\nDataset \t Dim. \t Mem. Usage \n')
        print(' PTrain  \t {} \t {:06.2f}\n'.format(self.dataset['train'].shape),
                                                   getsizeof(self.dataset['ptrain'])/(10**(-6)))
        print(' PTest   \t {} \t {:06.2f}\n'.format(self.dataset['test'].shape),
                                                   getsizeof(self.dataset['ptest']/(10**(-6)))

        # NOTE: by-pixel label masks -> Preprocessor -> ohe label for each image returned
        # NOTE: Preprocessed data is stored to avoid low-level (i.e pixel neighborhood)
        #       class imbalance problems during training. Test data is left imbalanced.
        # NOTE: The number of training instances to be used should be used by preprocessing()
        #       and should be specified in spec (e.g. spec = { ... 'ntrain':5000})

        # Run the model's graph
        self.results = self.evaluate_model()

    #def bootstrap(self):

    def evaluate_model(self):

        with tf.Session(graph = self.graph) as session:
            print('\nFitting model...')
            tf.global_variables_initializer().run()

            # Fit the model
            for step in range(spec['training_steps']):

                    batch_data, batch_labels = minibatch(self.dataset['ptrain'], self.labels['ptrain'],
                                                         self.spec['batch_size'], step)
                    fd = {tf_train_data:batch_data, tf_train_labels:batch_labels}
                    _, l, pred = session.run([optimizer, loss, tf_train_predictions], feed_dict = fd)

                    print('\n\nTraining accuracy @ step {}: {:04.2f}'.format(step, accuracy_score(pred, batch_labels)))
                    print('\nTraining loss @ step {}: {:06.2f}').format(step, l)

            # Evaluate the model's testing performance
            test_predictions = tf_test_predictions.eval(feed_dct = {tf_test_data:self.dataset['ptest']})
            test_accuracy    = accuracy_score(test_predictions, self.labels['ptest'])
            print('\n\nTesting accuracy @ step {}: {:04.2f}'.format(spec['training_steps'], test_accuracy))
            #test_predictions = mask_ohe(test_predictions, self.labels['train']) # Reshape to original mask dimensions
