import tensorflow as tf
import cnn_utils

"""
    Our model. Based on DeepMask architecture.
    
    Features:
        - 5 forward layers
        - 1 skip layer
        - 1 recurrent layer
"""

def variables():
    """
        Returns a dict of variables for use with .model()
    """
    
    v = { 'C1' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 1, in_channels = 1,  out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME'),
          'C2' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 8, out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = True),
          'C3' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 8, out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = True),
          'C4' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 8, out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = True),
          'C5' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 8, out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = True),
          'C6' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 40, out_channels = 8,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = False),
          'C7' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = 8, out_channels = 2, activate = False,
                                                             stride = [1, 1, 1, 1], padding = 'SAME', residual_layer = False)
        }
    return v
    
def graph(variables, images, label_masks, mode = 'training'):
    """
       The model's computational graph. It refers to a dict of variables (layer) that is declared later in the document.

       .variables() should be passed as the variables argument.
       images and labels are both tensors (images is a stack of images, labels is a matrix of one-hot-encoded labels).
       
       Returns the loss (a scalar) and predictions (a stack of one-channel images).
    """
    # Forward pass
    C1 = variables['C1'].apply(input_data = images, mode = mode)
    C2 = variables['C2'].apply(input_data = C1, mode = mode)
    C3 = variables['C3'].apply(input_data = C2, mode = mode)
    C4 = variables['C4'].apply(input_data = C3, mode = mode)
    C5 = variables['C5'].apply(input_data = C4, mode = mode)
    
    # Layer connected to all previous layers
    Concat1 = tf.concat([C1, C2, C3, C4, C5], axis = -1)
    C6 = variables['C6'].apply(input_data = Concat1, mode = mode)
    
    # Linear projection layer
    C7 = variables['C7'].apply(input_data = C6, mode = mode)
    
    logits = C7
    cross_entropy, predictions, iou = cnn_utils.get_loss_predictions_iou(logits, label_masks)
    
    return cross_entropy, predictions, iou