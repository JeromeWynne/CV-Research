##============= DESCRIPTION ===============##
"""
Implements a simple two-layer convolutional neural network.

Preprocessing:

Model:

Training policy:

"""
print('\nModel initialized. Importing libraries...')

##============= LIBRARIES =================##
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactive warnings about building from sourc
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import numpy as np

from models import two_layer_convnet
from preprocessing import preprocessing # Hand-crafted preprocessing function
from pymvtest import classification

##============= SCRIPT =================##
### >> DATA IMPORT << ##
print('Importing data...')
images = np.array([np.expand_dims(imread(fp), axis = -1)
			for fp in glob('../data/resized-images/*.png')])
masks  = np.array([imread(fp)/255
 			for fp in glob('../data/masked-images/*.png')])

TF = {
      'batch_size':32,
      'input_channels':1,
      'n_classes':2,
      'n_testing_images':1,
      'n_training_samples':5000,
      'patch_size':25,
      'image_size':250,
      'query_side':100,
      'output_channels':[64, 128, 1024],
      'filter_size':[3, 5],
      'seed':3,
      'mode':'holdout',
      'split_fraction':0.7,
      'training_steps':10001,
      'learning_rate':0.01,
      'graph':tf.Graph(),
      'summary_train':[None]*3, # Loss, accuracy, image
      'summary_test':[None]*3 # Query, predictions, filters
} # Master dictionary

print('{} training iterations.'.format(TF['training_steps']))
print('{} units per batch.'.format(TF['batch_size']))

TF = two_layer_convnet(TF)

tester = classification.Tester(dataset      = images,
                               masks        = masks,
                               TF           = TF,
                               preprocessor = preprocessing) # Tester applies preprocessing, runs graph according to spec.

tester.evaluate_model()
