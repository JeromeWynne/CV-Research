##============= DESCRIPTION ===============##
"""
Implements a simple two-layer convolutional neural network.

Preprocessing:

Model:

Training policy:

"""

##============= LIBRARIES =================##
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactive warnings about building from sourc
from scipy.misc import imread, imresize
from pymvtest import classification
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import numpy as np


##============= FUNCTIONS =================##
def extract_patches(images, masks, patch_size, mode):
    """
    	Extracts patches from image data.
	If training, returns a balanced set of patches.
    	If testing, returns patches for every pixel in every image passed.

    Args:
        images:     np.float32 array of images.
        masks:      np.float32 array of by-pixel class masks.
        patch_size: integer indicating kernel side length.
        train:      bool indicating whether to return a balanced dataset.

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

        ix['nocrack'] = ix['nocrack'][
				np.random.randint(low = 0,
                                                  high = ix['nocrack'].shape[0],
                                                  size = ix['crack'].shape[0]),
                                      :]
        ix = np.concatenate([ix['crack'], ix['nocrack']], axis = 0)
    # Testing mode does not balance dataset (i.e. extracts all pixels)
    elif mode == 'test':
        ix = np.array(np.nonzero(masks != None)).T

    patches    = np.zeros([ix.shape[0], patch_size, patch_size, 1])
    ohe_labels = np.zeros([ix.shape[0], 2])

    # 2. Sample patches centered on selected pixels
    for j, i in enumerate(np.random.permutation(ix)):

        if (j == 0) or np.any(images[i[0], :, :, :] != img):
            img  = images[i[0], :, :, :]
            pimg = np.pad(img, pad_width = [[ps, ps], [ps, ps], [0, 0]],
                          mode = 'constant', constant_values = 0)

        patches[j, :, :, :] =  pimg[i[0] + ps // 2 : i[0] + (3 * ps // 2),
                                    i[1] + ps // 2 : i[1] + (3 * ps // 2)]
        k = masks[i[0], i[1], i[2]]
        ohe_labels[j, int(k)] = 1. # One-hot encode the masks

    return patches, ohe_labels


def preprocessing(self, dataset, masks, mode):
    """
    Extracts and whitens patches. Designed in accordance with Tester requirements.

    Arguments:
        dataset: np.float32 array of images.
        labels:  np.float32 array of by-pixel masks for the images.
        train:   bool indicating whether to balance dataset and store filter parameters.

    Returns:
        filtered_dataset: subset images.
        filtered_labels:  one-hot encoded labels
    """
    # Resizing
    IMAGE_SIZE      = 200
    MASK_THRESHOLD  = 0.2
    resized_dataset = np.array([imresize(img.squeeze(), [IMAGE_SIZE, IMAGE_SIZE]) for img in dataset])
    resized_masks   = np.array([imresize(msk, [IMAGE_SIZE, IMAGE_SIZE]) for msk in masks])
    resized_masks   = np.greater(resized_masks, MASK_THRESHOLD*255)

    resized_dataset = np.expand_dims(resized_dataset, axis = -1)

    # Subsetting
    PATCH_SIZE     = 20

    # Filtering
    if mode == 'train':
        subset_dataset, subset_labels  = extract_patches(resized_dataset,
                                                         resized_masks, PATCH_SIZE, mode = 'train') # Balanced subset
        self.pp_parameters['mean'] = np.mean(subset_dataset, axis = 0)
        self.pp_parameters['std']  = np.std(subset_dataset, axis = 0)
        print('Training dataset dimensions: {}'.format(subset_dataset.shape))
        print('Training labels dimensions: {}'.format(subset_labels.shape))

    elif mode == 'valid':
        subset_dataset, subset_labels  = extract_patches(resized_dataset,
                                                         resized_masks, PATCH_SIZE, mode = 'valid')
        print('Validation dataset dimensions: {}'.format(subset_dataset.shape))
        print('Validation labels dimensions: {}'.format(subset_labels.shape))

    filtered_dataset = (subset_dataset.astype(np.float32)
                            - self.pp_parameters['mean'])/self.pp_parameters['std']
    filtered_labels  = subset_labels.astype(np.float32)

    return filtered_dataset, filtered_labels


##============= SCRIPT =================##
print('\nModel initialized.')
### >> DATA IMPORT << ##
images = np.array([np.expand_dims(imread(fp), axis = -1)
			for fp in glob('../data/resized-images/*.png')])
masks  = np.array([imread(fp)/255
 			for fp in glob('../data/masked-images/*.png')])
print('Data imported.')

### >> GRAPH DEFINITION << ###
TF = {
      'batch_size':32,
      'input_channels':1,
      'n_classes':2,
      'image_size':20,
      'output_channels':[64, 128, 1024],
      'filter_size':[3, 3],
      'seed':1,
      'mode':'holdout',
      'split_fraction':0.8,
      'training_steps':5001,
      'learning_rate':0.05,
      'graph':tf.Graph(),
      } # Master dictionary

with TF['graph'].as_default():
    # Placeholders and constants
    TF['training_data']   = tf.placeholder(tf.float32,
                                           [TF['batch_size'], TF['image_size'],
                                                TF['image_size'], TF['input_channels']],
                                            name = 'training_data')
    TF['training_labels'] = tf.placeholder(tf.float32,
                                           [TF['batch_size'], TF['n_classes']],
                                           name = 'training_labels')
    TF['validation_data'] = tf.placeholder(tf.float32,
                                            [None, TF['image_size'], TF['image_size'],
                                             TF['input_channels']],
                                            name = 'validation_data')
    TF['validation_labels'] = tf.placeholder(tf.float32,
                                            [None, TF{'n_classes'}],
                                            name = 'validation_labels')

    # Variables
    with tf.name_scope('Variables'):
        # Convolution layers
        with tf.name_scope('VariablesConvLayers'):
            filters1 = tf.Variable(tf.truncated_normal(
                    	           shape = [TF['filter_size'][0], TF['filter_size'][0],
                                            TF['input_channels'], TF['output_channels'][0]], stddev = 0.01),
			                       name = 'Layer_1_Filters')
            biases1  = tf.Variable(tf.zeros([TF['output_channels'][0]]),
			                       name = 'Layer_1_Biases')
            filters2 = tf.Variable(tf.truncated_normal(
                    	           shape = [TF['filter_size'][1], TF['filter_size'][1],
                                            TF['output_channels'][0], TF['output_channels'][1]], stddev = 0.01),
			                       name = 'Layer_2_Filters')
            biases2  = tf.Variable(tf.zeros([TF['output_channels'][1]]),
			                       name = 'Layer_2_Biases')
        # Fully connected layers
        with tf.name_scope('VariablesFCLayer'):
            weights3 = tf.Variable(tf.truncated_normal(
                                   shape = [TF['output_channels'][1] *
                                                TF['image_size'] *
                                                TF['image_size'],
                                                TF['n_classes']], stddev = 0.01),
			name = 'FC_Layer_Weights')
            biases3  = tf.Variable(tf.zeros(TF['n_classes']),
			name = 'FC_Layer_Biases')

    # Model
    def model(data):
        with tf.name_scope('ConvolutionalLayers'):
            # Layer 1 : 20 x 20 x 1 input ; 10 x 10 x 64 output ; (3 x 3 x 1) x 64 filters ; stride of 2 ; same padding
            conv = tf.nn.conv2d(data, filters1, strides = [1, 1, 1, 1],
			    padding = 'SAME', use_cudnn_on_gpu = True,
			    name = 'Layer_1_Conv')
            act  = tf.nn.relu(conv + biases1, name = 'Layer_1_Response')

            # Layer 2 : 10 x 10 x 64 input ; 5 x 5 x 128 output ; (3 x 3 x 64) x 128 filters ; stride of 2 ; same padding
            conv = tf.nn.conv2d(act, filters2, strides = [1, 1, 1, 1],
			    padding = 'SAME', use_cudnn_on_gpu = True,
			    name = 'Layer_2_Conv')
            act  = tf.nn.relu(conv + biases2, name = 'Layer_2_Response')

        with tf.name_scope('FullyConnectedLayer'):
            # Layer 3 : fully connected ; 5*5*128 input ; (5*5*128 x 2) filters
            shape  = tf.shape(act)
            act    = tf.reshape(act, [shape[0], shape[1]*shape[2]*shape[3]])
            logits = tf.nn.relu(tf.matmul(act, weights3) + biases3, name = 'FC_Layer_Logits')
            return logits

    # Loss and optimizer
    with tf.name_scope('Training'):

        logits = model(TF['training_data'])

        with tf.name_scope('Loss'):
            TF['loss'] = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                labels = TF['training_labels']),
                            name = 'loss')
            tf.summary.scalar('CrossEntropy', TF['loss'])

        with tf.name_scope('Optimizer'):
            TF['optimizer'] = tf.train.GradientDescentOptimizer(TF['learning_rate'],
                                    name = 'optimizer').minimize(TF['loss'])

        trp = tf.nn.softmax(logits)

    with tf.name_scope('Validation'):
        vp  = tf.nn.softmax(model(TF['validation_data']))

    # Predictions and Accuracy Scores
    TF['training_predictions'] = tf.identity(trp, name = 'train_predictions')
    TF['training_accuracy']    = tf.metrics.accuracy(TF['training_labels'], TF['training_predictions'],
                                                     name = 'training_accuracy')

    TF['validation_predictions']  = tf.identity(vp, name = 'validation_predictions')
    TF['validation_accuracy']     = tf.metrics.accuracy(TF['validation_labels'], TF['validation_predictions'],
                                                        name = 'validation_accuracy')

    tf.summary.scalar('TrainingAccuracy', TF['training_accuracy'])
    tf.summary.scalar('ValidationAccuracy', TF['validation_accuracy'])

    TF['summary'] = tf.summary.merge_all()

print('Graph defined.')

### >> TRAINING << ###

print('{} training iterations.'.format(TF['training_steps']))
print('{} units per batch.'.format(TF['batch_size']))

tester = classification.Tester(dataset      = images,
                               masks        = masks,
                               TF           = TF,
                               preprocessor = preprocessing) # Tester applies preprocessing, runs graph according to spec.

tester.evaluate_model()
