import tensorflow as tf
import numpy as np

class ConvLayer(object):
    def __init__(self, map_size, in_channels, out_channels):
        self.filters    = tf.Variable(tf.truncated_normal(
                                        shape = [map_size, map_size, in_channels, out_channels],
                                        stddev = 0.1))
        self.biases     = tf.Variable(tf.zeros(shape = [out_channels]))

    def training(self, data_in):
        self.conv       = tf.nn.conv2d(data_in, self.filters, strides = [1, 1, 1, 1],
                                       padding = 'SAME', use_cudnn_on_gpu = True)
        return tf.nn.relu(self.conv + self.biases)

    def testing(self, data_in):
        self.conv       = tf.nn.conv2d(data_in, self.filters, strides = [1, 1, 1, 1],
                                       padding = 'SAME', use_cudnn_on_gpu = True)
        return tf.nn.relu(self.conv + self.biases)

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out):
        self.weights = tf.Variable(tf.truncated_normal(shape = [n_in, n_out],
                                                       stddev = 0.01))
        self.biases = tf.Variable(tf.zeros(shape = [n_out]))

    def training(self, data_in):
        shape = tf.shape(data_in)
        reshaped_data_in = tf.reshape(data_in, [shape[0], shape[1]*shape[2]*shape[3]])
        projection  = tf.matmul(reshaped_data_in, self.weights) + self.biases
        return projection

    def testing(self, data_in):
        shape = tf.shape(data_in)
        reshaped_data_in = tf.reshape(data_in, [shape[0], shape[1]*shape[2]*shape[3]])
        projection  = tf.matmul(reshaped_data_in, self.weights) + self.biases
        return projection


def model(TF):

    TF['graph'] = tf.Graph()

    with TF['graph'].as_default():

        with tf.name_scope('Inputs'):
            TF['data']   = tf.placeholder(tf.float32,
                                         [None, TF['patch_size'],
                                         TF['patch_size'], TF['input_channels']],
                                        name = 'data')
            TF['labels'] = tf.placeholder(tf.float32,
                                          [None, TF['n_classes']],
                                          name = 'labels')
            TF['summary_test'].append(tf.summary.image('ValidationImages',
                                            tf.slice(TF['data'], [0, 0, 0, 0],
                                                     [-1, -1, -1, 1])))

        with tf.name_scope('ConvolutionalLayers'):
            Layer1 = ConvLayer(TF['filter_size'][0], TF['input_channels'],
                               TF['output_channels'][0])
            Layer2 = ConvLayer(TF['filter_size'][1], TF['output_channels'][0],
                               TF['output_channels'][1])

        with tf.name_scope('FullyConnectedLayer'):
            Layer3 = FullyConnectedLayer(TF['output_channels'][1]*TF['patch_size']*TF['patch_size'],
                                         TF['n_classes'])

        def training_model(data_in):
            activation1 = Layer1.training(data_in)
            activation2 = Layer2.training(activation1)
            projection  = Layer3.training(activation2)

            TF['summary_train'].append(tf.summary.histogram('FirstLayerFilters',
                                                            Layer1.filters))
            TF['summary_train'].append(tf.summary.histogram('SecondLayerFilters',
                                                            Layer2.filters))
            TF['summary_train'].append(tf.summary.histogram('ThirdLayerWeights',
                                                            Layer3.weights))
            return projection

        def testing_model(data_in):
            activation1 = Layer1.testing(data_in)
            activation2 = Layer2.testing(activation1)
            projection  = Layer3.testing(activation2)
            return projection

        with tf.name_scope('Training'):
            training_logits         = training_model(TF['data'])
            TF['loss']              = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                        logits = training_logits,
                                                        labels = TF['labels']))
            TF['optimize']          = tf.train.GradientDescentOptimizer(
                                                TF['learning_rate']).minimize(TF['loss'])
            TF['training_predictions'] = tf.nn.softmax(training_logits)
            TF['training_accuracy'] = tf.contrib.metrics.accuracy(
                                            tf.argmax(TF['labels'], axis = 1),
                                            tf.argmax(TF['training_predictions'], axis = 1))
            TF['summary_train'].append(tf.summary.scalar('TrainingAccuracy',
                                                         TF['training_accuracy']))

        with tf.name_scope('Testing'):
            testing_logits            = testing_model(TF['data'])
            TF['testing_predictions'] = tf.nn.softmax(testing_logits)
            TF['testing_accuracy']    = tf.contrib.metrics.accuracy(
                                                tf.argmax(TF['labels'], axis = 1),
                                                tf.argmax(TF['testing_predictions'], axis = 1))
            TF['summary_test'].append(tf.summary.scalar('ValidationAccuracy',
                                                        TF['testing_accuracy']))

        TF['saver'] = tf.train.Saver()
        TF['test_summary']  = tf.summary.merge(TF['summary_test'])
        TF['train_summary'] = tf.summary.merge(TF['summary_train'])
        return TF
