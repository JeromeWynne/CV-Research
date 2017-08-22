import tensorflow as tf

# The idea here is to:
# > Define a model.
# > Import training data.
# > Use the data to fit the model.


# Dataset API
# - Allows us to handle large amounts of data, different data formats, and complicated transformations.
# - tf.contrib.data.Dataset -> A sequence of elements: Each element contains one or more tensors (e.g. example pairs).
    # We can create a dataset by:
        # . Dataset.from_tensor_slices() constructs a dataset from a list of Tensors
        # . Apply a transformation to build a dataset from another Dataset.
        
# - tf.contrib.data.Iterator -> Allows us to extract elements from a dataset.
    # Iterator.get_next() retrieves the next element from a Dataset when run.
    # Iterator.initializer allows us to reinitialize an iteror, so we can iterate over a dataset multiple times.
    

# Basic Mechanics

# What does our pipeline look like?
# tf.contrib.data.Dataset.from_tensors() -> Creates a dataset from tensors
# How do we create a Dataset from images in memory? I guess we can make a numpy array, then make a Tensor from that.

# We can pass Dataset objects to various methods to transform it
# -> Dataset.map() applies a transformation to each element
# -> Dataset.batch() applies multiple-element transformations 

tf.contrib.data offers an easy way to construct efficient pipelines.

# The elements of a Dataset can be subsets of tensors - these tensors are known as components.
# Each component:
    # - Has a tf.DType which represents the element types.
    # - tf.TensorShape represents the shape of the tensors.
    # - Dataset.output_types and Dataset.output_shapes allow us 
    #   to look at what the types and shapes of a component are.
    
# Example: Dataset creation from tensors.
datasetA = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
datasetB = tf.contrib.data.Dataset.from_tensor_slices(
             { 'OneComponent' : tf.random_uniform([4]),
               'TwoComponent' : tf.random_uniform([4, 100], maxval = 100, dtyp = tf.float32) }
             ) # We can name the components of a Dataset

# Example: Dataset transformations.
# The transformations applied to the dataset are applied on an element-by-element basis.
# We apply a transformation using a function that takes as many arguments as there are components
# to an element. Let's do an example.

datasetA_2 = datasetA.map(lambda a: a**2)
datasetB_2 = datasetB.flat_map(lambda x, y: (x**2, y + 2)) # How do we return multiple values from a lambda?


# To iterate through our dataset, we can create an Iterator. There are three kinds of iterator:
    # one-shot
    # initializable
    # reinitializable
    # feedable
    # (3?)
    
# What is the difference between tf.contrib.data.Dataset.from_tensor() and ... .from_tensor_slices() ?

dataset      = tf.contrib.data.Dataset.range(100)
iterator     = dataset.make_one_shot_iterator() # Associate an iterator with the dataset.

for i in range(100):
    value = sess.run(iterator.get_next()) # Obviously we need a session object to execute this.
    assert i == value

    
# An initializable iterator requires

training_dataset   = tf.contrib.data.Dataset.range(100).map(
                           lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)

iterator = Iterator.from_structure(training_dataset.output_types, 
                                   training_dataset.output_shapes)

training_init_op   = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

for _ in range(50):
    sess.run(training_init_op) # Attaches the iterator to the training dataset
    for _ in range(100):
        sess.run(iterator.get_next()) # So I guess this gets one element
        
    sess.run(validation_init_op) # Attaches the iterator to the validation dataset
    for _ in range(50):
        sess.run(iterator.get_next())

        
# We can run an iterator until exhaustion as follows:
while True:
    try:
        sess.run(iterator.get_next())
    except tf.errors.OutOfRangeError:
        break
        
        
        
        
        
        
# 1. Import training and validation datasets + masks as numpy arrays.
# 2. Verify their ranges.
# 3. Retain the specified number of training examples.
# 4. Specify a graph for the model to be trained.
#    This accepts a minibatch for each optimization step.
#    Use Dataset.batch(k) to make size-k batches of data.
#    The model diagnostics are also contained within this graph.

# 5. Pass the training and validation datasets into a graph.
#    So we need one tensor for the entire dataset, then individual ones for training operations?

# 6. Apply preprocessing to the training data.
# 7. Use minibatch gradient descent to fit the model using the training data.
# 8. Apply the fit model to the validation data.


import numpy as np
from scipy.misc import imread
from glob import glob


TRAINING_EPOCHS = 10
tf.set_random_seed(42)
np.random.seed(42)

data = { 'training_images'  : np.array([imread(fp, mode = 'L') for fp in glob('./Data/170822_Panels/training/images/*.png')]),
         'training_masks'   : np.array([imread(fp, mode = 'L') for fp in glob('./Data/170822_Panels/training/masks/*.png')]),
         'validation_images': np.array([imread(fp, mode = 'L') for fp in glob('./Data/170822_Panels/validation/images/*.png')]),
         'validation_masks' : np.array([imread(fp, mode = 'L') for fp in glob('./Data/170822_Panels/validation/masks/*.png')])
       } # These are all in [0, 255]

graph = tf.Graph()

with graph.as_default():

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
    
    # Apply training dataset augmentations
    training_dataset = training_dataset.repeat(TRAINING_EPOCHS)
    
    def augment_image(image_tensor, mask_tensor, mode):
        """
            image_tensor is [k, k, 1]
            mask_tensor is  [k, k, 1]
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
    validation_dataset = training_dataset.map(lambda q, gt: augment_image(q, gt, mode = 'validation'))