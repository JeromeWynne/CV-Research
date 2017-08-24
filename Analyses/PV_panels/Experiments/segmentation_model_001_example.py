import tensorflow as tf
import cnn_utils

def variables():
    """
        Returns a dict of variables for use with example_model_001.model()
    """
    
    # TOMORROW: WE NEED TO CHANGE THE WAY BATCHNORMALIZEDCONVLAYER IS CREATED
    
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

    output['loss'], output['predictions'], output['meanIoU'] = cnn_utils.get_loss_predictions_iou(output['logits'], masks)
    # NOTE: IMAGES CONTAINING NO CRACK ARE DISCARDED FOR THE meanIoU CALCULATION

    return output['loss'], output['predictions'], output['meanIoU']