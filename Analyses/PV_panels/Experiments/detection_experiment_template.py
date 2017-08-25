from cnn_utils import augment_image
from scipy.misc import imread
from shutil import rmtree
import tensorflow as tf
from glob import glob
from os import mkdir
import numpy as np

"""
    Before running this script, make sure that:
        1. A ./Results directory has been created.
        2. A ./SavedModels directory has been created.
        3. The DATA_DIR has been set correctly.
        4. The MODEL_ID has been updated to reflect the chosen segmentation model.
"""

""" Configuration """
TRAINING_EPOCHS = 10
MINIBATCH_SIZE  = 32
INITIAL_LEARNING_RATE = 0.01
MODEL_ID        = "001_DetectionExample"
DATA_DIR        = './data/pv_panels/'
import detection_model_001_example as detection_model

np.random.seed(42)


""" Numpy data import """
print('Reading data...')
data = { 'training_images'  : np.array([imread(fp, mode = 'L') for fp in glob(DATA_DIR + 'training/images/*.png')]),
         'training_masks'   : np.array([imread(fp, mode = 'L') for fp in glob(DATA_DIR + 'training/masks/*.png')]),
         'validation_images': np.array([imread(fp, mode = 'L') for fp in glob(DATA_DIR + 'validation/images/*.png')]),
         'validation_masks' : np.array([imread(fp, mode = 'L') for fp in glob(DATA_DIR + 'validation/masks/*.png')])
       } # These are all in [0, 255]


""" Computational graph 

    1. Import the dataset - all images are the same size.
    2. Apply preprocessing to the training data - contrast shift, random flip and transpose.
    3. One-hot encode the label masks.
    4. Declare the model variables (these are defined in segmentation_model).
    5. Add the model's operations to the graph - the model receives the
       images + their label masks, and returns the loss and IoU scores, along with the predicted segmentations.
    6. We include summary operations to track the queries, ground truth (i.e. label masks), predictions, mean IoU,
       and loss. These contitute our key experimental metrics - we can visualize them using TensorBoard.
"""

print('Data read.\nDefining graph...')
graph = tf.Graph()

with graph.as_default():
    
    tf.set_random_seed(42)
    
    """ Graph - TensorFlow data import """
    with tf.name_scope('Import'):
        training_dataset = tf.contrib.data.Dataset.from_tensor_slices(
                                    ( data['training_images'][:, :, :, np.newaxis].astype(np.float32),
                                      np.greater(np.sum(data['training_masks'], axis = [1, 2]), 1).astype(np.float32) )
                                    )

        validation_dataset = tf.contrib.data.Dataset.from_tensor_slices(
                                    ( data['validation_images'][:, :, :, np.newaxis].astype(np.float32),
                                      np.greater(np.sum(data['validation_masks'], axis = [1, 2]), 1.).astype(np.float32) )
                                    ) # Note that the labels are not one-hot encoded on import
        
    """ Graph - Preprocessing """
    with tf.name_scope('Preprocessing'):

        training_dataset   = training_dataset.repeat(TRAINING_EPOCHS) # the training dataset is not balanced, nor is the validation set.
        training_dataset   = training_dataset.map(lambda q, gt: augment_image(q, gt, mode = 'training')) # one-hot encoding is here
        training_dataset   = training_dataset.batch(MINIBATCH_SIZE)
        
        validation_dataset = validation_dataset.map(lambda q, gt: augment_image(q, gt, mode = 'validation'))
        validation_dataset = validation_dataset.batch(data['validation_images'].shape[0])
    
    with tf.name_scope('IteratorInitialization'): # The iterators must be manually initialized before the dataset can be iterated over
        training_iterator   = tf.contrib.data.Iterator.from_dataset(training_dataset)
        validation_iterator = tf.contrib.data.Iterator.from_dataset(validation_dataset)
    
    """ Graph - Model Variables """
    with tf.name_scope('Variables'):
        training_step = tf.Variable(0, trainable = False)
        variables     = detection_model.variables()
    
    """ Graph - Predictions and Optimization """
    with tf.name_scope('Training'):
        training_images, training_labels = training_iterator.get_next()
        training_loss, training_predictions, training_accuracy = detection_model.graph(variables, training_images,
                                                                                       training_labels, mode = 'training')
        
        training_summary = tf.summary.merge([
                                tf.summary.scalar('MeanCrossEntropy', training_loss),
                                tf.summary.scalar('Accuracy',         training_accuracy),
                                tf.summary.image( 'Predictions',      training_predictions, max_outputs = 1),
                                tf.summary.image( 'Queries',          training_images,      max_outputs = 1),
                                tf.summary.image( 'GroundTruth',      
                                                  tf.slice(training_labels, begin = [0, 0],size = [-1, 1]),
                                                  max_outputs = 1)
                             ])
        
        with tf.name_scope('Optimization'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = INITIAL_LEARNING_RATE)
            gradients = optimizer.compute_gradients(training_loss)
            optimize  = optimizer.apply_gradients(gradients, global_step = training_step)
        
    with tf.name_scope('Validation'):
        validation_images, validation_labels = validation_iterator.get_next()
        validation_loss, validation_predictions, validation_accuracy = segmentation_model.graph(variables, validation_images,
                                                                                                validation_labels, mode = 'validation')
        validation_summary = tf.summary.merge([
                                tf.summary.scalar('MeanCrossEntropy', validation_loss),
                                tf.summary.scalar('Accuracy', validation_accuracy),
                                tf.summary.image( 'Predictions',      validation_predictions, max_outputs = 100),
                                tf.summary.image( 'Queries',          validation_images,      max_outputs = 100),
                                tf.summary.image( 'GroundTruth',
                                                  tf.slice(validation_labels, begin = [0, 0], size = [-1, 1]),
                                                  max_outputs = 100)
                             ])
    
    saver = tf.train.Saver()
    


"""
    Training & Testing

        1. Initialize the summary writers.
        2. Initialize the variables and iterators.
        4. Run the train operation until the iterator throws a tf.errors.OutOfRangeError
        5. When validating (we only validate after the full training session):
            - Initialize the validation set iterator.
            - Run the validation loss, predictions, iou operations.
        6. Save the model parameters, close the writers.
"""
print('Graph defined.\nInitializing session...')
with tf.Session(graph = graph) as session:
    try:
        mkdir('./results/' + MODEL_ID)
        mkdir('./results/' + MODEL_ID + '/training')
        mkdir('./results/' + MODEL_ID + '/validation')
        
    except OSError:
        rmtree('./results/' + MODEL_ID)
        mkdir('./results/'  + MODEL_ID)
        mkdir('./results/' + MODEL_ID + '/training')
        mkdir('./results/' + MODEL_ID + '/validation')
    
    training_writer   = tf.summary.FileWriter('./results/' + MODEL_ID + '/training', session.graph)
    validation_writer = tf.summary.FileWriter('./results/' + MODEL_ID + '/validation', session.graph)
    
    session.run([tf.global_variables_initializer(),
                 training_iterator.initializer, validation_iterator.initializer])
    print('Session initialized. \nTraining for {} epochs.'.format(TRAINING_EPOCHS))
    
    while True:
        try:
            _, step_summary = session.run([optimize, training_summary])
            step_number     = tf.train.global_step(session, training_step)
            training_writer.add_summary(step_summary, step_number)
            
        except tf.errors.OutOfRangeError: # When we reach the end of the training dataset, test the model
            validation_summary = session.run(validation_summary)
            validation_writer.add_summary(validation_summary, step_number)
            break
            
    training_writer.close()
    validation_writer.close()
    saver.save(session, './saved_models/' + MODEL_ID)
    print('\n Experiment complete.')