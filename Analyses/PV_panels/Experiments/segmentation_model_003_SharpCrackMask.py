import tensorflow as tf
import cnn_utils

"""
    Our model. Based on DeepMask architecture.
    
    Features:
        - 5 forward layers
        - 5 skip layers
        - 1 recurrent layer
"""

def variables():
    """
        Returns a dict of variables for use with .model()
    """
    
    N_CHANNELS = 7
    
    # Default stride = [1, 1, 1, 1], padding = 'SAME'
    v = { 
          'F1' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 1, in_channels = 1,  out_channels = N_CHANNELS),
        
          'F2' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = N_CHANNELS,  out_channels = N_CHANNELS),
        
          'S2' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = N_CHANNELS, out_channels = N_CHANNELS,
                                                             residual_layer = True),
        
          'M2' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 3, in_channels = N_CHANNELS*2, out_channels = N_CHANNELS,
                                                             residual_layer = False),
         
          'N' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 5, in_channels = N_CHANNELS, out_channels = N_CHANNELS,
                                                            residual_layer = False),
        
          'H' : cnn_utils.BatchNormalizedConvolutionalLayer(map_size = 1, in_channels = N_CHANNELS, out_channels = 2,
                                                            residual_layer = False, activate = False)
        }
    
    v['F5'] = v['F4'] = v['F3'] = v['F2']
    v['S5'] = v['S4'] = v['S3'] = v['S2']
    v['M5'] = v['M4'] = v['M3'] = v['M2']
    
    return v
    
def graph(variables, images, label_masks, mode = 'training'):
    """
       The model's computational graph. It refers to a dict of variables (layer) that is declared later in the document.

       .variables() should be passed as the variables argument.
       images and labels are both tensors (images is a stack of images, labels is a matrix of one-hot-encoded labels).
       
       Returns the loss (a scalar) and predictions (a stack of one-channel images).
    """
    # Forward pass
    F1 = variables['F1'].apply(input_data = images, mode = mode)
    F2 = variables['F2'].apply(input_data = F1, mode = mode)
    F3 = variables['F3'].apply(input_data = F2, mode = mode)
    F4 = variables['F4'].apply(input_data = F3, mode = mode)
    F5 = variables['F5'].apply(input_data = F4, mode = mode)
    
    # Skip connections
    S2 = variables['S2'].apply(input_data = F2, mode = mode)
    S3 = variables['S3'].apply(input_data = F3, mode = mode)
    S4 = variables['S4'].apply(input_data = F4, mode = mode)
    S5 = variables['S5'].apply(input_data = F5, mode = mode)
    
    # Backward connections
    M5 = variables['M5'].apply(input_data = tf.concat([S5, F5], axis = -1), mode = mode)
    M4 = variables['M4'].apply(input_data = tf.concat([S4, M5], axis = -1), mode = mode)
    M3 = variables['M3'].apply(input_data = tf.concat([S3, M4], axis = -1), mode = mode)
    M2 = variables['M2'].apply(input_data = tf.concat([S2, M3], axis = -1), mode = mode)
    
    # Neck
    N = variables['N'].apply(input_data = M2, mode = mode)
    
    # Linear projection layer
    H = variables['H'].apply(input_data = N, mode = mode)
    
    logits = H
    cross_entropy, predictions, iou = cnn_utils.get_loss_predictions_iou(logits, label_masks)
    
    return cross_entropy, predictions, iou