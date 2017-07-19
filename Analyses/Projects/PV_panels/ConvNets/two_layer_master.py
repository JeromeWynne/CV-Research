
print('\nMaster initialized.\nImporting libraries...')

##============= LIBRARIES =================##
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactive warnings about building from sourc
from scipy.misc import imread, imresize
from glob import glob
import numpy as np

from models import two_layer_convnet
from preprocessors import subset_and_whiten # Hand-crafted preprocessing function
from pymvtest import classification

##============= SCRIPT =================##
### >> DATA IMPORT << ##
print('Importing data...')
images = np.array([np.expand_dims(imread(fp), axis = -1)
			for fp in glob('./data/resized-images/*.png')])
masks  = np.array([imread(fp)/255
 			for fp in glob('./data/masked-images/*.png')])

TF = {
      'batch_size':32,
      'filter_size':[3, 5],
      'graph':None,
      'image_size':250, # Resize full images to this size
      'input_channels':1,
      'learning_rate':0.01,
      'n_classes':2,
      'n_testing_images':1,
      'n_training_samples':5000,
      'output_channels':[64, 128, 1024],
      'patch_size':25,
      'query_side':100, # Size of contiguous test query image
      'seed':3,
      'split_fraction':0.7,
      'summary_train':[None]*3, # Loss, accuracy, image
      'summary_test':[None]*3, # Query, predictions, filters
      'training_steps':10001,
	  'test_id':'two_layer_convnet_img250_ptch25_pxint'
} # Master dictionary

print('{} training iterations.'.format(TF['training_steps']))
print('{} units per batch.'.format(TF['batch_size']))

TF = two_layer_convnet.model(TF)

tester = classification.Tester(dataset      = images,
                               masks        = masks,
                               TF           = TF,
                               preprocessor = subset_and_whiten.preprocessing
                            )
tester.evaluate_model()
