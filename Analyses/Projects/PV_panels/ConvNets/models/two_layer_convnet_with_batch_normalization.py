import tensorflow as tf
import numpy as np

def model(TF):
    TF['graph'] = tf.Graph()

    with TF['graph'].as_default():
        # Placeholders and constants
        with tf.name_scope('Inputs'):
            TF['data']   = tf.placeholder(tf.float32,
                                         [None, TF['patch_size'],
                                         TF['patch_size'], TF['input_channels']],
                                        name = 'data')
            TF['labels'] = tf.placeholder(tf.float32,
                                          [None, TF['n_classes']],
                                          name = 'labels')
            TF['summary_train'].append(tf.summary.image('BatchImages',
                                       tf.slice(TF['data'], [0, 0, 0, 0],
                                        [-1, -1, -1, 1]), max_outputs = 1))

        # Variables
        with tf.name_scope('Variables'):
            # Convolution layers
            with tf.name_scope('VariablesConvLayers'):
                filters1 = tf.Variable(tf.truncated_normal(
                        	           shape = [TF['filter_size'][0], TF['filter_size'][0],
                                                TF['input_channels'], TF['output_channels'][0]], stddev = 0.01),
        		                       name = 'Layer_1_Filters')
                biases1  = tf.Variable(tf.zeros([TF['output_channels'][0]]),
       		                           name = 'Layer_1_Biases')
                filters2 = tf.Variable(tf.truncated_normal(
                        	           shape = [TF['filter_size'][1], TF['filter_size'][1],
                                                TF['output_channels'][0], TF['output_channels'][1]], stddev = 0.01),
        		                       name = 'Layer_2_Filters')
                biases2  = tf.Variable(tf.zeros([TF['output_channels'][1]]),
        		                       name = 'Layer_2_Biases')

                TF['summary_train'].append(tf.summary.image('FirstLayerFilters',
                                           tf.slice(tf.transpose(filters1, [3, 0, 1, 2]),
                                                    [0, 0, 0, 0], [-1, -1, -1, 1]),
                                                    max_outputs = 16))
                TF['summary_train'].append(tf.summary.histogram('FirstLayerFilters', filters1))
                TF['summary_train'].append(tf.summary.histogram('SecondLayerFilters',
                                           filters2))
            # Fully connected layers
            with tf.name_scope('VariablesFCLayer'):
                weights3 = tf.Variable(tf.truncated_normal(
                                       shape = [TF['output_channels'][1] *
                                                    TF['patch_size'] *
                                                    TF['patch_size'],
                                                    TF['n_classes']], stddev = 0.01),
        		                       name = 'FC_Layer_Weights')
                biases3  = tf.Variable(tf.zeros(TF['n_classes']),
        		                       name = 'FC_Layer_Biases')
                TF['summary_train'].append(tf.summary.histogram('FCLayerWeights', weights3))

        # Model
        def model(data):
            with tf.name_scope('ConvolutionalLayers'):
                # Layer 1 : 20 x 20 x 1 input ; 10 x 10 x 64 output ; (3 x 3 x 1) x 64 filters ; stride of 2 ; same padding
                conv1 = tf.nn.conv2d(data, filters1, strides = [1, 1, 1, 1],
        		                     padding = 'SAME', use_cudnn_on_gpu = True,
        		                     name = 'Layer_1_Conv')
                c1_mean, c1_variance = tf.nn.moments(conv1, axes = [0, 1, 2], keep_dims = False)
                bn1   = tf.nn.batch_normalization(conv1, mean = c1_mean,
                                                  variance = c1_variance,
                                                  offset = None, scale = None,
                                                  variance_epsilon = 1e-5) # offset is beta, scale is gamma
                TF['summary_train'].append(tf.summary.histogram('BatchNormalizedOutputs1', bn1))
                act1  = tf.nn.relu(bn1 + biases1, name = 'Layer_1_Response')

                # Layer 2 : 10 x 10 x 64 input ; 5 x 5 x 128 output ; (3 x 3 x 64) x 128 filters ; stride of 2 ; same padding
                conv2 = tf.nn.conv2d(act1, filters2, strides = [1, 1, 1, 1],
            		                 padding = 'SAME', use_cudnn_on_gpu = True,
        		                     name = 'Layer_2_Conv')
                c2_mean, c2_variance = tf.nn.moments(conv2, axes = [0, 1, 2], keep_dims = False)
                bn2   = tf.nn.batch_normalization(conv2, mean = c2_mean,
                                                  variance = c2_variance,
                                                  offset = None, scale = None,
                                                  variance_epsilon = 1e-5) # offset is beta, scale is gamma
                TF['summary_train'].append(tf.summary.histogram('BatchNormalizedOutputs2', bn2))
                act2  = tf.nn.relu(bn2 + biases2, name = 'Layer_2_Response')

            with tf.name_scope('FullyConnectedLayer'):
                # Layer 3 : fully connected ; 5*5*128 input ; (5*5*128 x 2) filters
                shape  = tf.shape(act2)
                act2    = tf.reshape(act2, [shape[0], shape[1]*shape[2]*shape[3]])
                logits = tf.nn.relu(tf.matmul(act2, weights3) + biases3, name = 'FC_Layer_Logits')

            return logits

        # Loss and optimizer
        logits = model(TF['data'])

        with tf.name_scope('Training'):
            with tf.name_scope('Loss'):
                TF['loss'] = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                    labels = TF['labels']),
                                name = 'loss')
                TF['summary_train'].append(tf.summary.scalar('CrossEntropy', TF['loss']))

            with tf.name_scope('Optimizer'):
                TF['optimizer'] = tf.train.GradientDescentOptimizer(TF['learning_rate'],
                                        name = 'optimizer').minimize(TF['loss'])

            # Predictions and Accuracy Scores
            TF['predictions'] = tf.nn.softmax(logits, name = 'predictions')

            TF['accuracy']    = tf.contrib.metrics.accuracy(tf.argmax(TF['labels'], axis = 1),
                                                            tf.argmax(TF['predictions'], axis = 1))
            TF['summary_train'].append(tf.summary.scalar('Accuracy', TF['accuracy']))

        TF['train_summary'] = tf.summary.merge(TF['summary_train'])

        TF['saver'] = tf.train.Saver()

    return TF
