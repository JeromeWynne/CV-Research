import tensorflow as tf
import cnn_utils

"""
    A replication of the crack-detection model described in http://onlinelibrary.wiley.com/doi/10.1111/mice.12263/abstract.
    
    Deviations from the original material:
        - Images fed are grayscale, not RGB.
"""

def variables():
    """
        Returns a dict of variables for use with .model()
    """
    
    v = { 'C1' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 20, in_channels = 1,  out_channels = 24,
                                                             stride = [1, 2, 2, 1], padding = 'VALID'),
          'C2' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 15, in_channels = 24, out_channels = 48,
                                                             stride = [1, 2, 2, 1], padding = 'VALID'),
          'C3' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 10, in_channels = 48, out_channels = 96,
                                                             stride = [1, 2, 2, 1], padding = 'VALID'),
          'FC4' : cnn_utils.ConvolutionalLayer(map_size = 1, in_channels = 96, out_channels = 96,
                                               padding = 'VALID') # Fully-connected layer 1
          'FC5' : cnn_utils.ConvolutionalLayer(map_size = 1, in_channels = 96, out_channels = 2,
                                               padding = 'VALID', activate = False) # Fully-connected layer 2
        }
    return v
    
def graph(variables, images, labels, mode = 'training'):
    """
       The model's computational graph. It refers to a dict of variables (layer) that is declared later in the document.

       .variables() should be passed as the variables argument.
       images and labels are both tensors (images is a stack of images, labels is a matrix of one-hot-encoded labels).
       
       Returns the loss (a scalar) and predictions (a stack of one-channel images).
    """
    C1     = variables['C1'].apply(input_data = images, mode = mode)
    P1     = tf.nn.pool(C1, window_shape = [1, 7, 7, 1],
                        pooling_type = 'MAX', padding = 'SAME')
    
    C2     = variables['C2'].apply(input_data = P1, mode = mode)
    P2     = tf.nn.pool(C3, window_shape = [1, 4, 4, 1],
                        pooling_type = 'MAX', padding = 'SAME')
    
    C3     = variables['C3'].apply(input_data = P2, mode = mode)
    FC4    = variables['FC4'].apply(input_data = C3, mode = mode)
    FC5    = variables['FC5'].apply(input_data = FC4, mode = mode)
    
    logits = tf.squeeze(FC5)
    cross_entropy, predictions, accuracy = cnn_utils.get_loss_predictions_accuracy(logits, labels)
    
    return cross_entropy, predictions, accuracy