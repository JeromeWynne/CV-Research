from types import MethodType
from os.path import exists
import tensorflow as tf
import numpy as np


def augment_images(images, scores):
    """
        Applies augmentations to a minibatch.
        > Flips
        > Whitens
    """
    # Flip images lr
    flip_lr = np.random.randint(0, 2, images.shape[0])
    images[flip_lr, :, :, :] = np.flip(images[flip_lr, :, :, :], axis = 2)
    scores[flip_lr, :, :, :] = np.flip(scores[flip_lr, :, :, :], axis = 2)
    
    # Flip images ud
    flip_ud = np.random.randint(0, 2, images.shape[0])
    images[flip_ud, :, :, :] = np.flip(images[flip_ud, :, :, :], axis = 1)
    scores[flip_ud, :, :, :] = np.flip(scores[flip_ud, :, :, :], axis = 1)
    
    # Gamma shift
    images      = images + 1
    gamma_shift = np.random.randint(0, 2, images.shape[0])
    gamma       = np.min([np.max([0.7, np.random.normal(1, scale = 0.2)]), 1.3])
    images[gamma_shift, :, :, :] = images[gamma_shift, :, :, :]**gamma
    images[gamma_shift, :, :, :] = images[gamma_shift, :, :, :]/images[gamma_shift, :, :, :].max()
    images = images - 1
    
    return images, scores

def get_minibatch(batch_size, images, scores, labels):
    """
        Returns a balanced and augmented subset of the images and scores passed.
    
        Labels is a binary mask indicating which images contain cracks
        Images is a stack of images, ordered in the same sequence as scores
        Scores is a stack of np.float32 masks of crack scores, ordered in the same sequence as images
    """
    ix = {
          'cracked'   : np.array(np.nonzero(labels)).squeeze(),
          'uncracked' : np.array(np.nonzero(np.logical_not(labels))).squeeze()
         }
    ix['cracked']    = np.random.choice(ix['cracked'],
                                     size = batch_size // 2, replace = True)
    ix['uncracked']  = np.random.choice(ix['uncracked'],
                                       size = batch_size // 2, replace = True)
    ix               = list(ix['cracked']) + list(ix['uncracked'])

    minibatch_images, minibatch_scores = augment_images(images[ix, :, :, :],
                                                        scores[ix, :, :, :])
    return minibatch_images, minibatch_scores

class CNNModel:
    
    """
        Initializes an empty network.
        
        1. Initialize a CNNModel instance, specifying the learning rate and number of input channels.
        2. Define the variables that you need by grabbing the graph
        3. Re-define the function 'model'. It must take mode = 'training', and mode = 'testing'.
            > Access the input data using self.ground_truth.
            > The final layer should return an minibatch_size x image_h x image_w x 2 output.
            > To append summaries, use tf.summary.merge([tf.training_summary, other_summary])
    """
    
    def __init__(self, define_variables, compute_graph, learning_rate = 1., model_id = '0'):
        self.graph            = tf.Graph()
        self.learning_rate    = learning_rate
        self.model_id         = model_id
        self.model            = MethodType(compute_graph, self)
        self.define_variables = MethodType(define_variables, self)
        self.ops              = {}
        self.lam              = 10.
        
    def compile_graph(self):
        with self.graph.as_default():
            with tf.name_scope('Inputs'):
                self.images       = tf.placeholder(tf.float32, shape = [None, None, None, None])
                self.ground_truth = tf.placeholder(tf.float32, shape = [None, None, None, 2])
                self.global_step  = tf.Variable(0, trainable = False)
            
            with tf.name_scope('Variables'):
                self.variables    = self.define_variables()

            with tf.name_scope('Training'):
                logits                     = self.model(mode = 'train', variables = self.variables)
                activated                  = tf.nn.softmax(logits, dim = -1)
                self.training_predictions  = tf.slice(activated, [0, 0, 0, 0], [-1, -1, -1, 1])
                penalty_kernel             = tf.constant([[-1, -1, -1],
                                                          [-1, 9, -1],
                                                          [-1, -1, -1]], dtype=tf.float32)/9.
                penalty_kernel             = tf.expand_dims(tf.expand_dims(penalty_kernel, -1), -1)
                incompatibility            = tf.nn.conv2d(2*self.training_predictions - 1, penalty_kernel,
                                                            strides = [1, 1, 1, 1],
                                                            padding = 'SAME', use_cudnn_on_gpu = True)
                flat_incompatibility       = tf.reshape(tf.abs(incompatibility), [-1])
                ohe_logits                 = tf.reshape(logits, [-1, 2])
                ohe_scores                 = tf.reshape(self.ground_truth, [-1, 2])
                ce                         = tf.nn.softmax_cross_entropy_with_logits(
                                                    labels = ohe_scores, logits = ohe_logits)
                self.reg_incompatibility   = self.lam*tf.reduce_mean(flat_incompatibility*ce)
                self.training_loss         = self.reg_incompatibility + tf.reduce_mean(ce)

                with tf.name_scope('Optimization'):
                    optimizer      = tf.train.AdadeltaOptimizer(self.learning_rate)
                    gradients      = optimizer.compute_gradients(self.training_loss, tf.trainable_variables())
                    self.optimize  = optimizer.apply_gradients(gradients, global_step = self.global_step)

                    self.training_summary = tf.summary.merge(
                                             [ tf.summary.scalar('Loss', self.training_loss),
                                               tf.summary.scalar('MeanRegIncompatibility', self.reg_incompatibility),
                                               tf.summary.image('Images', self.images),
                                               tf.summary.image('GroundTruth',
                                                                tf.slice(self.ground_truth, [0, 0, 0, 0], [-1, -1, -1, 1])),
                                               tf.summary.image('Predictions', self.training_predictions)]
                                            )
                
            with tf.name_scope('Testing'):
                logits                    = self.model(mode = 'test', variables = self.variables)
                activated                 = tf.nn.softmax(logits, dim = -1)
                self.testing_predictions  = tf.slice(activated, [0, 0, 0, 0], [-1, -1, -1, 1])
                ohe_logits                = tf.reshape(logits, [-1, 2])
                ohe_scores                = tf.reshape(self.ground_truth, [-1, 2])
                self.testing_loss         = tf.reduce_mean(
                                                tf.nn.softmax_cross_entropy_with_logits(
                                                    labels = ohe_scores, logits = ohe_logits))

                self.testing_summary = tf.summary.merge(
                                         [ tf.summary.scalar('Loss', self.testing_loss),
                                           tf.summary.image('Images', self.images),
                                           tf.summary.image('GroundTruth',
                                                            tf.slice(self.ground_truth, [0, 0, 0, 0], [-1, -1, -1, 1])),
                                           tf.summary.image('Predictions', self.testing_predictions) ]
                                        )

            self.saver = tf.train.Saver()
            
            
    def train(self, training_images, training_scores, training_labels,
              query_images, query_scores, training_steps=10001, batch_size=16,
              clear_previous=True, minibatch_period = 100, validation_period = 500):
        
        with tf.Session(graph = self.graph) as session:

            if exists('./Checkpoints/' + self.model_id + '.index') and not clear_previous:
                print('Restoring model...')
                self.saver.restore(session, './Checkpoints/' + self.model_id)

                print(' Model restored. Continuing to train...\n\n')
            else:
                print('Training initialized.\n\n')
                session.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter('./Results/' + self.model_id + '/train', session.graph)
            test_writer  = tf.summary.FileWriter('./Results/' + self.model_id + '/test', session.graph)

            for _ in range(training_steps):
                minibatch_images, minibatch_scores = get_minibatch(batch_size, training_images,
                                                                   training_scores, training_labels)
                fd = { self.images        : minibatch_images,
                       self.ground_truth  : minibatch_scores }
                _, l, summary, step = session.run([ self.optimize, self.training_loss,
                                                    self.training_summary, self.global_step ], feed_dict = fd)
                train_writer.add_summary(summary, step)

                if step % minibatch_period == 0:
                    print('Mean minibatch cross-entropy @ step {:^5d}: {:>3.3f}'.format(step, l))

                if step % validation_period == 0:
                    fd   = { self.images       : query_images,
                             self.ground_truth : query_scores }
                    summary, step = session.run([self.testing_summary,
                                                 self.global_step ], feed_dict = fd)
                    test_writer.add_summary(summary, step)

            self.saver.save(session, './Checkpoints/' + self.model_id)
            train_writer.close()
            test_writer.close()
            
            
class BatchNormalizedConvLayer(object):
    def __init__(self, map_size, in_channels, out_channels, activate=True, padding = 'SAME'):
        if type(map_size) != list: map_size = [map_size, map_size]
        self.filters    = tf.Variable(tf.truncated_normal(
                                        shape = [map_size[0], map_size[1], in_channels, out_channels],
                                        stddev = 0.01))
        
        self.biases     = tf.Variable(tf.zeros(shape = [out_channels]))
        self.input_mean = tf.Variable(tf.zeros(shape = [out_channels]), trainable = False)
        self.input_var  = tf.Variable(tf.zeros(shape = [out_channels]), trainable = False)
        self.decay      = 0.999
        self.activate   = activate
        self.padding    = padding

    def apply(self, mode, data_in):
        
        self.conv        = tf.nn.conv2d(data_in, self.filters, strides = [1, 1, 1, 1],
                                       padding = self.padding, use_cudnn_on_gpu = True)
        mu, var          = tf.nn.moments(self.conv, axes = [0, 1, 2], keep_dims = False)
        
        self.update_mean = tf.assign(self.input_mean, self.decay*(self.input_mean)
                                                      + (1 - self.decay)*mu)
        self.update_var  = tf.assign(self.input_var,  self.decay*(self.input_var)
                                                      + (1 - self.decay)*var)
        
        if mode == 'train':
            with tf.control_dependencies([self.update_mean, self.update_var]):
                self.norm_conv  = tf.nn.batch_normalization(self.conv, mean = mu,
                                                            variance = var, offset = None,
                                                            scale = None, variance_epsilon = 1e-5)
        if mode == 'test':
                self.norm_conv  = tf.nn.batch_normalization(self.conv, mean = mu,
                                                variance = var, offset = None,
                                                scale = None, variance_epsilon = 1e-5)
            
        if self.activate:
            return tf.nn.relu(self.norm_conv + self.biases)
        else:
            return self.norm_conv + self.biases