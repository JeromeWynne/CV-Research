
print('\nMaster initialized.\nImporting libraries...')

##============= LIBRARIES =================##
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Deactive warnings about building from sourc
from scipy.misc import imread, imresize
from glob import glob
import numpy as np
from datetime import datetime

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

dt = datetime.now()

TF = {
      'batch_size':32,
      'filter_size':[3, 5],
      'graph':None,
      'image_size':200, # Resize full images to this size
      'patch_size':20,
      'input_channels':1,
      'learning_rate':0.001,
      'n_classes':2,
      'n_testing_images':1,
      'n_training_samples':5000,
      'output_channels':[16, 8, 32],
      'seed':3,
      'split_fraction':0.7,
      'summary_train':[None]*6, # Loss, accuracy, image
      'summary_test':[None]*3, # Query, predictions, filters
      'training_steps':20001,
	  'test_id':str(dt.year)+str(dt.month)+str(dt.day)+str(dt.minute)+str(dt.second),
	  'subdir':'image_and_patch_size',
	  'model_id':'two_layer_convnet',
	  'preprocessor_id':'subset_and_whiten'
} # Master dictionary

print('{} training iterations.'.format(TF['training_steps']))
print('{} units per batch.'.format(TF['batch_size']))

patch_sizes = [5, 10, 20, 40]
image_sizes = [50, 75, 100, 200]

for ps in patch_sizes:
	for i in image_sizes:
		TF['image_size'] = i
		TF['patch_size'] = ps
		TF['query_side'] = i//2
		TF['test_id']    = '1_imgs_'+str(i)+'_patchs_'+str(ps)
		TF = two_layer_convnet.model(TF)
		tester = classification.Tester(dataset      = images,
		                               masks        = masks,
		                               TF           = TF,
		                               preprocessor = subset_and_whiten.preprocessing
		                            )
		tester.evaluate_model()
