"""
    cnn_utils.py
    
    This package provides objects for building convolutional neural networks using TensorFlow.
    
    Classes:
        BatchNormalizedConvLayer
        InceptionLayer
        
    Methods:
        flatten_ohe_tensors
        get_loss_predictions_iou
"""

from numpy.random import randint
import tensorflow as tf
            
class BatchNormalizedConvLayer(object):
    def __init__(self, map_size, in_channels, out_channels,
                 activate=True, padding = 'SAME', residual_layer = False):
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
        self.residual   = residual_layer

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
            if self.residual: return tf.nn.relu(self.norm_conv + self.biases) + data_in
            else:             return tf.nn.relu(self.norm_conv + self.biases)
        else:
            if self.residual: return self.norm_conv + self.biases + data_in
            else:             return self.norm_conv + self.biases
            
            
class InceptionLayer:
    """
        Returns a batch-normalized, residual Inception layer, the set of mappings F = {1x1 -> 3x3, 1x1 -> 5x5, 1x1}
        
        Residual implementation demands that the number of input and output channels are the same.
    
        > Optimal network topology is found by clustering the highly correlated outputs of
          previous layers.
        > Low-level layers tend to have channel-wise correlations.
        > Deeper layers will be correlated spatially.
        > 1x1 convolutions are effectively embeddings of channel-wise representation.
        > We cluster these embeddings in our Inception layer.
    """
    
    def __init__(self, in_channels, out_channels,
                 bottleneck_channels, n_3x3_out, n_5x5_out,
                 residual_layer = True):
        
        if (n_3x3_out + n_5x5_out > out_channels):
            raise ValueError("The number of output channels must be greater than the sum of its constituents.")
        
        self.in_channels         = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.residual_layer      = residual_layer
        self.n_1x1_out   = out_channels - (n_3x3_out + n_5x5_out)
        self.n_3x3_out   = n_3x3_out
        self.n_5x5_out   = n_5x5_out
        self.F1x1 = BatchNormalizedConvLayer(map_size = 1,
                                             in_channels = self.in_channels,
                                             out_channels = self.n_1x1_out)
        self.B1x1_3 = BatchNormalizedConvLayer(map_size = 1,
                                               in_channels = self.in_channels,
                                               out_channels = self.bottleneck_channels)
        
        self.B1x1_5 = BatchNormalizedConvLayer(map_size = 1,
                                               in_channels  = self.in_channels,
                                               out_channels = self.bottleneck_channels)
        self.F3x3   = BatchNormalizedConvLayer(map_size = 3,
                                               in_channels  = self.bottleneck_channels,
                                               out_channels = self.n_3x3_out)
        self.F5x5   = BatchNormalizedConvLayer(map_size = 5,
                                               in_channels  = self.bottleneck_channels,
                                               out_channels = self.n_5x5_out)
        
    def apply(self, mode, data_in):
        F1x1_out   = self.F1x1.apply(mode, data_in)
        B1x1_3_out = self.B1x1_3.apply(mode, data_in)
        B1x1_5_out = self.B1x1_5.apply(mode, data_in)
        F3x3_out   = self.F3x3.apply(mode, B1x1_3_out)
        F5x5_out   = self.F5x5.apply(mode, B1x1_5_out)
        
        overall_output = tf.concat([F1x1_out, F3x3_out, F5x5_out], axis = -1)
        
        if self.residual_layer:
            return overall_output + data_in
        else:
            return overall_output
        
        
def flatten_ohe_tensors(tensors):
    """
            tensors is a list of OHE tensors to be flattened.

            Returns N x 2 matrices for one-hot-encoded image-shaped mask and prediction tensors.
        """
    flattened_tensors = [None]*len(tensors)

    for i, tensor in enumerate(tensors):
        flattened_tensors[i] = tf.reshape(tensor, [-1, 2])

    return flattened_tensors

def get_loss_predictions_iou(logits, masks):
    """
        Accepts image-shaped logits and ground truth masks.
        
        Returns scalar mean cross-entropy, scalar mean intersection-over-union (IoU), and image-shaped predictions.
    """
    
    flat_logits, flat_masks = cnn_utils.flatten_ohe_tensors([logits, masks])
    loss                    = tf.nn.softmax_cross_entropy_with_logits(labels = flat_masks,
                                                                      logits = flat_logits)
    predictions = tf.slice(tf.nn.softmax(logits, dim = -1),
                           begin = [0, 0, 0, 0], size = [-1, -1, -1, 1])
    labels      = tf.greater(tf.slice(masks, dim = -1,
                                      begin = [0, 0, 0, 0], size = [-1, -1, -1, 1]),
                             0.5) # Pixel class labels (in a stack of image-shaped matrices)
    intersection = tf.reduce_sum(tf.logical_and(tf.greater(predictions, 0.5), labels),
                                 axis = [1, 2, 3])
    union        = tf.reduce_sum(tf.logical_or(tf.greater(predictions, 0.5), labels),
                                 axis = [1, 2, 3])
    IoU          = tf.where(tf.less(union, 1e-7), tf.zeros_like(union), # Only calculate IoU for images containing cracks
                                                  tf.divide(intersection, union))
    meanIoU      = tf.divide(tf.reduce_sum(IoU), tf.count_nonzero(IoU))
                            
    return loss, predictions, meanIoU
                            
def augment_image(image_tensor, mask_tensor, mode):
    """
                        image_tensor is [k, k, 1]
                        mask_tensor is  [k, k, 1]

                        Returns (image_tensor, mask_tensor) according to training/validation transformations.
                        mask_tensor is one-hot encoded.
             """
    image_tensor = tf.image.per_image_standardization(image_tensor) # mean zero, unit s.d.
    mask_tensor  = mask_tensor/255 # {0., 1.}

    if mode == 'training':
        image_tensor = tf.image.random_flip_left_right(image_tensor)
        image_tensor = tf.image.random_flip_up_down(image_tensor)
        image_tensor = tf.image.random_contrast(image_tensor, lower = 0.5, upper = 1.5)
        if randint(0, 2):
            image_tensor = tf.image.transpose_image(image_tensor)

    mask_tensor = tf.stack([mask_tensor, 1 - mask_tensor], axis = -1)
    return image_tensor, mask_tensor
