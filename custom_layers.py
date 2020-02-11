# Custom layers for Tensorflow:
# - GlobalConv1D: 1D convolution that is invariant wrt permutations of the features
#                 Every convolutional kernel has two, one for the center (the value)
#                 the kernel is centered on, and one for the surround (all other values)
# - TopKPool: selects the top-k values (equal to Maxpooling for k=1)

# (c) matthias treder 2019
import numpy as np
import re, math

import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.keras import models, layers, Model
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, Conv1D, MaxPool1D
from tensorflow.python.keras import backend
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils

class GlobalConv1D(Layer):

    def __init__(self, filters, activation='linear',  kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', use_bias=True, **kwargs):
        self.filters = filters
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias

        super(GlobalConv1D, self).__init__(**kwargs)

    def wrap_around_1d_kernel(self, width, n_input_chans):
        '''Creates a convolutional kernel that wraps-around the features. It
        has a central weight. Surround weights are all the same. In order to have a
        wrap-around functionality, the true size of the filter will be 2*width + 1'''

        #c =  tf.get_variable("center", [1, n_input_chans, n_output_chans], initializer=self.initializer)
        #s =  tf.get_variable("surround", [1,n_input_chans, n_output_chans], initializer=self.initializer)

        # Create weight for center of convolutional kernel
        c = self.add_weight(name='kernel',
                                      shape=(1, n_input_chans, self.filters),
        #                                      shape=(1, n_input_chans, tf.convert_to_tensor(self.filters)),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        # Create weight for surround of convolutional kernel
        s = self.add_weight(name='surround',
                                   shape=(1, n_input_chans, self.filters),
                                    initializer=self.kernel_initializer,
                                    trainable=True)

        return tf.keras.backend.concatenate([s]*width + [c] + [s]*width, axis=0)

    def build(self, input_shape):

        self.need_to_expand_dims = len(input_shape)<3
        n_input_chans = 1 if len(input_shape)<3 else input_shape[2].value # .value: casts shape to int

        #print('N_INPUT CHANS: ', n_input_chans)
        # Create wrap-around convolutional kernel
        self.kernel = self.wrap_around_1d_kernel(input_shape[1], n_input_chans)
        #print('expand dims: ', self.need_to_expand_dims)

        # Create bias weights (one bias term per output chan, since all the
        # input chans get combined during convolution)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                    shape=(1, 1, self.filters),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        super(GlobalConv1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        #print('x: ', x)
        #print('kernel: ', self.kernel)
        # Create convolution operation
        if self.need_to_expand_dims:
            # If this layer comes directly after the input layer, and the input
            # shape was given as e.g. (nfeatures, ) it is just one-dimensional,
            # so we need to expand the dimensions
            op = tf.nn.conv1d(tf.expand_dims(x,-1), self.kernel, stride=1, padding='SAME') # data_format='NCW')
        else:
            # If this layer comes not as first layer, or the input chans have been
            # explicitly defined as e.g. (nfeatures, 1) it is already two-dimensional,
            # so we do not need to expand the dimensions
            op = tf.nn.conv1d(x, self.kernel, stride=1, padding='SAME') # data_format='NCW')

        #print('op: ', op)
        #print('bias: ', self.bias)
        if self.use_bias: op += self.bias

        #return tf.matmul(x,x)
        #out = K.dot(x,self.kernel)
        #out = K.dot(x, x)

        #return conv

        # Apply activation function and return
        return tf.keras.layers.Activation(self.activation)(op)
        #return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



def TopKPool(k):
    '''Pooling layer that retrieves the top K values from
    every input channel.'''
    # x needs to be transposed since top k operates on the last dimension
    # which is channels in most cases. If the top K layer comes directly after
    # the input layer and it is one-dimensional e.g. shape = (n_features,) then
    # the input layer needs to be changed such that the second dimension (channels)
    # is specified as shape = (n_features,1)
    # After the operation, it is transposed back so that the rows correspond to
    # channels again
    fcn = lambda x: tf.convert_to_tensor(tf.linalg.transpose(tf.math.top_k(tf.linalg.transpose(x),k)[0]))
    return Lambda(fcn, trainable=False, name="top_{}_pool".format(k))
