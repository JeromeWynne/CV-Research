import example_model_001 as model
from scipy.misc import imread
import tensorflow as tf
from glob import glob
import numpy as np


""" Configuration """
TRAINING_EPOCHS = 10
MINIBATCH_SIZE  = 32

tf.set_random_seed(42)
np.random.seed(42)


""" Numpy data import """
data_dir = './Data/170822_Panels/'
data = { 'training_images'  : np.array([imread(fp, mode = 'L') for fp in glob(data_dir + 'training/images/*.png')]),
         'training_masks'   : np.array([imread(fp, mode = 'L') for fp in glob(data_dir + 'training/masks/*.png')]),
         'validation_images': np.array([imread(fp, mode = 'L') for fp in glob(data_dir + 'validation/images/*.png')]),
       } # These are all in [0, 255]


""" Computational graph """
graph = tf.Graph()

with graph.as_default():
    
    """ Graph - TensorFlow data import """
    with tf.name_space('Import'):
        training_dataset = tf.contrib.data.Dataset.from_tensor_slices(
                                    {
                                     'Queries'    : data['training_images'][:, :, :, np.newaxis].astype(np.float32),
                                     'GroundTruth': data['training_masks'][:, :, :, np.newaxis].astype(np.float32)
                                    })

        validation_dataset = tf.contrib.data.Dataset.from_tensor_slices(
                                    {
                                     'Queries'    : data['validation_images'][:, :, :, np.newaxis].astype(np.float32),
                                     'GroundTruth': data['validation_masks'][:, :, :, np.newaxis].astype(np.float32)
                                    })
        
    """ Graph - Preprocessing """
    with tf.name_space('Preprocessing'):

        # Apply training dataset augmentations
        training_dataset = training_dataset.repeat(TRAINING_EPOCHS)
        
        def augment_image(image_tensor, mask_tensor, mode):
            """
                        image_tensor is [k, k, 1]
                        mask_tensor is  [k, k, 1]

                        Returns (image_tensor, mask_tensor) according to training/validation transformations.
                        mask_tensor is one-hot encoded.
             """
            image_tensor = tf.image.per_image_standardization(image_tensor) # mean zero, unit s.d.
            mask_tensor  = mask_tensor/255 # {0., 1.}

            if mode == 'training':
                image_tensor = tf.image.random_flip_left_right(image_tensor)
                image_tensor = tf.image.random_flip_up_down(image_tensor)
                image_tensor = tf.image.random_contrast(image_tensor, lower = 0.5, upper = 1.5)
                if np.random.randint(0, 2):
                    image_tensor = tf.image.transpose_image(image_tensor)

            mask_tensor = tf.stack([mask_tensor, 1 - mask_tensor], axis = -1)
            return image_tensor, mask_tensor

        training_dataset   = training_dataset.map(lambda q, gt: augment_image(q, gt, mode = 'training'))
        training_dataset   = training_dataset.batch(MINIBATCH_SIZE)
        validation_dataset = training_dataset.map(lambda q, gt: augment_image(q, gt, mode = 'validation'))
    
    with tf.name_space('IteratorInitialization'): # The iterators must be manually initialized before the dataset can be iterated over
        training_iterator   = tf.contrib.data.Iterator.from_dataset(training_dataset)
        validation_iterator = tf.contrib.data.Iterator.from_dataset(validation_dataset)
    
    
    """ Graph - Model Variables """
    with tf.name_space('Variables'):
        training_step = tf.Variable(0, trainable = False)
        variables = model.variables()
    
    """ Graph - Predictions and Optimization """
    with tf.name_space('Training'):
        training_images, training_masks = training_iterator.get_next()
        training_loss, training_predictions = model.graph(variables, training_images,
                                                          training_masks, mode = 'training')
        
        with tf.name_space('Optimization'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = INITIAL_LEARNING_RATE)
            gradients = optimizer.compute_gradients(training_loss)
            train     = optimizer.apply_gradients(gradients, global_step = training_step)
        
    with tf.name_space('Validation'):
        validation_images, validation_masks = validation_iterator.get_next()
        validation_loss, validation_predictions = model.graph(variables, validation_images,
                                                              validation_masks, mode = 'validation')
    