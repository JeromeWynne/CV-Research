
print('\nMaster initialized.\nImporting libraries...')

##============= LIBRARIES =================##
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactive warnings about building from sourc
from scipy.misc import imread, imresize
from glob import glob
import numpy as np
from datetime import datetime

from models import two_layer_convnet
from pymvtest import classification

##============= SCRIPT =================##
TF = {
      'batch_size':32,
      'graph':None,
      'image_size':200, # Resize full images to this size
      'patch_size':20,
      'input_channels':1,
      'learning_rate':0.001,
      'n_classes':2,
      'output_channels':[16, 8, 32],
      'filter_size':[3, 5],
      'seed':3,
      'split_fraction':0.7,
      'summary_train':[],
      'training_steps':5001,
	  'test_id':'New_Build',
} # Master dictionary
print('Training for {} iterations.'.format(TF['training_steps']))
print('{} units per batch.'.format(TF['batch_size']))

### >> DATA IMPORT << ##
print('Importing data...')
images = np.array([np.expand_dims((imread(fp) - 128)/255, axis = -1)
			for fp in glob('./data/resized-images/*.png')])
masks  = np.array([imread(fp)/255
 			for fp in glob('./data/masked-images/*.png')])
images, masks = classification.resize(images, masks, TF['image_size'])

### >> FIT MODEL << ###
TF = two_layer_convnet.model(TF)
tester = classification.Tester(images, masks, TF)
tester.fit_model()
# tester.query_model()
