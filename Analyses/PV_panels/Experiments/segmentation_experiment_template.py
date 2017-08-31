from cnn_utils import augment_image, balance_dataset
from scipy.misc import imread
from shutil import rmtree
import tensorflow as tf
from glob import glob
from os import mkdir
import pandas as pd
import numpy as np

"""
    Before running this script, make sure that:
        1. A ./results directory has been created.
        2. A ./saved_models directory has been created.
        3. The DATA_DIR has been set correctly.
        4. The MODEL_ID has been updated to reflect the chosen segmentation model.
"""

""" Configuration """
TRAINING_EPOCHS = 30
MINIBATCH_SIZE  = 32
INITIAL_LEARNING_RATE = 0.1
DECAY_STEPS     = 20
DECAY_RATE      = 0.95
MODEL_ID        = "002_DeepMask"
DATA_DIR        = './data/pv_panels/'
import segmentation_model_002_CrackMask as segmentation_model

np.random.seed(42)


""" Import data filepaths """
print('Reading data...')
names = { 
    'training'   : [fp for fp in pd.read_csv(DATA_DIR + '/training/training_scheme.csv')['name']],
    'validation' : [fp for fp in pd.read_csv(DATA_DIR + '/validation/validation_scheme.csv')['name']]
}

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
        
    def get_batch(names, mode ,epochs):
        """
            Accepts a dictionary; names = {'training':..., 'validation':...} that contains lists of image names.
                
            Returns an op that fetches a batch of images and their one-hot encoded masks.
            
            (images, masks) -> queue #1 -> augmentation -> queue #2 -> minibatches of (images, masks)
        
            1. image_queue and mask_queue contain queues of image and mask filepaths, respectively.
            2. The WholeFileReader().read() method reads these filepaths and returns a string containing the file contents.
            3. The decode_png() method converts this string into uint image tensor.
            4. augment_image() standardizes the images ( + flips, rotates, and contrast-shifts them if mode == 'training').
               The spatial augmentations are also applied to the masks. The masks are one-hot encoded, meaning that they
               are returned as two-channel images. All tensors returned have dtype float32.
            5. shuffle_batch() forms another queue that allows us to randomly batch our images from a pool of size min_after_dequeue.
        
        """

        image_queue = tf.train.string_input_producer(
                                            [DATA_DIR + '/' + mode + '/images/' + n for n in names[mode]], 
                                                shuffle = False, num_epochs = epochs, name = mode + '_image_queue')
        mask_queue  = tf.train.string_input_producer(
                                            [DATA_DIR + '/' + mode + '/masks/' + n for n in names[mode]], 
                                                shuffle = False, num_epochs = epochs)
            
        reader = tf.WholeFileReader()
            
        _, image_string = reader.read(image_queue)
        _, mask_string  = reader.read(mask_queue)
           
        image = tf.image.decode_png(contents = image_string, channels = 1)
        mask  = tf.image.decode_png(contents = mask_string, channels = 1)
            
        image, mask = augment_image(image, mask, mode = mode)
        image_batch, mask_batch = tf.train.shuffle_batch([image, mask],
                                                          batch_size = MINIBATCH_SIZE,
                                                          capacity   = MINIBATCH_SIZE*13,
                                                          min_after_dequeue = MINIBATCH_SIZE*10,
                                                          shapes = [[185, 185, 1], [185, 185, 2]])
        return image_batch, mask_batch
        
    
    """ Graph - Model Variables """
    with tf.name_scope('Variables'):
        training_step = tf.Variable(0, trainable = False)
        variables     = segmentation_model.variables()
    
    """ Graph - Predictions and Optimization """
    with tf.name_scope('Training'):
        training_images, training_masks = get_batch(names, mode = 'training',  epochs = TRAINING_EPOCHS)
        training_loss, training_predictions, training_iou = segmentation_model.graph(variables, training_images,
                                                                                     training_masks, mode = 'training')
        
        with tf.name_scope('Optimization'):
            #learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step = training_step,
            #                                           decay_steps = DECAY_STEPS, decay_rate = DECAY_RATE)
            #optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss = training_loss,
            #                                                                     global_step = training_step)
            optimize = tf.train.AdadeltaOptimizer(1.).minimize(loss = training_loss,
                                                               global_step = training_step)
        
        training_summary = tf.summary.merge([
            tf.summary.scalar('MeanCrossEntropy', training_loss),
            tf.summary.scalar('MeanIoU',          training_iou),
            tf.summary.image( 'Predictions',      training_predictions, max_outputs = 1),
            tf.summary.image( 'Queries',          training_images,      max_outputs = 1),
            tf.summary.image( 'GroundTruth',      
                             tf.slice(training_masks, begin = [0, 0, 0, 0], size = [-1, -1, -1, 1]),
                             max_outputs = 1),
            tf.get_default_graph().get_tensor_by_name("Training/training_image_queue/fraction_of_32_full:0"),
            tf.get_default_graph().get_tensor_by_name("Training/input_producer/fraction_of_32_full:0")
            #tf.summary.scalar('LearningRate', learning_rate)
        ])
        
    with tf.name_scope('Validation'):
        validation_images, validation_masks = get_batch(names, mode = 'validation', epochs = 1)
        validation_loss, validation_predictions, validation_iou = segmentation_model.graph(variables, validation_images,
                                                                                           validation_masks, mode = 'validation')
        validation_summary = tf.summary.merge([
                                tf.summary.scalar('MeanCrossEntropy', validation_loss),
                                tf.summary.scalar('MeanIoU',          validation_iou),
                                tf.summary.image( 'Predictions',      validation_predictions, max_outputs = 100),
                                tf.summary.image( 'Queries',          validation_images,      max_outputs = 100),
                                tf.summary.image( 'GroundTruth',
                                                 tf.slice(validation_masks, begin = [0, 0, 0, 0],size = [-1, -1, -1, 1]),
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

validation_step = 0

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
    
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    
    coordinator = tf.train.Coordinator()
    threads     = tf.train.start_queue_runners(sess = session, coord = coordinator)
                                               
    print('Session initialized. \nTraining for {} epochs.'.format(TRAINING_EPOCHS))
    step_number = 0
    while True:
        try:
            if step_number % 100 == 1: # Record session performance every 100 steps
                run_metadata = tf.RunMetadata()
                _, step_summary = session.run([optimize, training_summary],
                                               options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                               run_metadata = run_metadata)
                step_number     = tf.train.global_step(session, training_step)
                training_writer.add_run_metadata(run_metadata, 'Step_{}'.format(step_number))
                training_writer.add_summary(step_summary, step_number)
            else:
                _, step_summary = session.run([optimize, training_summary])
                step_number     = tf.train.global_step(session, training_step)
                training_writer.add_summary(step_summary, step_number)
            
        except tf.errors.OutOfRangeError: # When we reach the end of the training dataset, test the model
            print('Training complete.\nValidating model...')
            try:
                validation_summary = session.run(validation_summary)
                validation_step += 1
                validation_writer.add_summary(validation_summary, validation_step)
                    
            except tf.errors.OutOfRangeError:
                print('Model validated. Experiment complete.')
                coordinator.request_stop() # Close the threads
                break
            break
            
    coordinator.join(threads) # What does this do?
    training_writer.close()
    validation_writer.close()
    saver.save(session, './saved_models/' + MODEL_ID)
    print('\n Experiment complete.')