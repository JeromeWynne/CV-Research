import tensorflow as tf
import cnn_utils

def _flatten_ohe_tensors(tensors):
    """
            tensors is a list of OHE tensors to be flattened.

            Returns N x 2 matrices for one-hot-encoded image-shaped mask and prediction tensors.
        """
    flattened_tensors = [None]*len(tensors)

    for i, tensor in enumerate(tensors):
        flattened_tensors[i] = tf.reshape(tensor, [-1, 2])

    return flattened_tensors

def variables():
    """
        Returns a dict of variables for use with example_model_001.model()
    """
    v = { 'F0_0' : cnn_utils.BatchNormalizedConvLayer(map_size = 1, input_channels = 1, output_channels = 8),
          'F1_0' : cnn_utils.BatchNormalizedConvLayer(map_size = 3, input_channels = 8, output_channels = 16),
          'O2_0' : cnn_utils.BatchNormalizedConvLayer(map_size = 1, input_channels = 16, output_channels = 2, activate = False)
        }
    return v
    
def graph(variables, images, masks, mode = 'training'):
    """
       The model's computational graph. It refers to a dict of variables (layer) that is declared later in the document.

       example_model_001.variables() should be passed as the variables argument.
       images and masks are both tensors.
       
       Returns the loss (a scalar) and predictions (a stack of one-channel images).
    """
    output           = {}
    output['F0_0']   = variables['F0_0'].apply(input_data = images,         mode = mode) # 1 x 1 x 1 conv x8
    output['F1_0']   = variables['F1_0'].apply(input_data = output['F0_0'], mode = mode) # 3 x 3 x 8 conv x16
    output['logits'] = variables['O2_0'].apply(input_data = output['F1_0'], mode = mode) # Linear projection: 16 -> 2 channels

    output['flat_logits'], output['flat_masks'] = _flatten_ohe_tensors([output['logits'], masks])
    output['loss']        = tf.nn.softmax_cross_entropy_with_logits(labels = output['flat_logits'],
                                                                    logits = output['flat_masks'])
    output['predictions'] = tf.slice(tf.nn.softmax(output['logits'], dim = -1),
                                     begin = [0, 0, 0, 0], size = [-1, -1, -1, 1])
    output['labels']        = tf.cast(tf.slice(masks, dim = -1,
                                               begin = [0, 0, 0, 0], size = [-1, -1, -1, 1]),
                                      dtype = tf.bool)
    output['intersection'] = tf.logical_and(tf.greater(output['predictions'], 0.5), output['labels'])
    output['union']        = tf.logical_or(tf.greater(output['predictions'], 0.5), output['labels'])
    output['IoU']          = tf.divide(tf.reduce_sum(output['intersection'], axis = [1, 2]),
                                       tf.reduce_sum(output['union'], axis = [1, 2])) # And for crackless images?

    return output['loss'], output['predictions'], output['IoU']