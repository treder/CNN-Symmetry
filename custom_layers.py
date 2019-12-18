# Custom layers for Tensorflow:
# - SymmetricConv2D: extension of Conv2D that adds weight sharing between pairs of
#                    filters (horizontal or vertical reflection symmetry)
# - GlobalConv1D: 1D convolution that is invariant wrt permutations of the features
#                 Every convolutional kernel has two, one for the center (the value)
#                 the kernel is centered on, and one for the surround (all other values)
# - TopKPool: selects the top-k values (equal to Maxpooling for k=1)

# (c) matthias treder 2019
import numpy as np
import re, math

import tensorflow as tf
from tensorflow.keras import models, layers, Model
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, Conv1D, MaxPool1D
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils

class SymmetricConv2D(layers.Conv2D):

    def __init__(self, filters, kernel_size,  symmetry={}, share_bias=True, **kwargs):
        '''
        symmetry - dict, number of filter flipped horizontally, vertically, or about both axes
        share_bias - if True, the biases of symmetric filters are also shared
        '''
        super(SymmetricConv2D, self).__init__(filters, kernel_size, **kwargs)

        # Set defaults for symmetric filters pairs
        self.symmetry = symmetry
        self.symmetry.setdefault('h', '30%')
        self.symmetry.setdefault('v', '20%')
        self.symmetry.setdefault('hv', '20%')
        self.share_bias = share_bias

        # If symmetric filters are specified as percentages, calculate the
        # absolute numbers here
        for key, val in self.symmetry.items():
            if (type(val) is str) and (val[-1]=='%'):
                val = round(float(val[:-1]) * self.filters / 100)
                # Make sure number of filters is divisible by 2 or 4
                if key in ['h','v']:
                    val = (val // 2) * 2
                elif key=='hv':
                    val = (val // 4) * 4
                self.symmetry[key] = val

        # Check if number of filters is divisible by 2 resp. 4
        for key, val in self.symmetry.items():
                if (key in ['h','v']) and (val % 2 != 0):
                    raise ValueError('Number of symmetric h and v filters must be divisible by 2')
                elif (key=='hv') and (val % 4 != 0):
                    raise ValueError('Number of symmetric hv filters must be divisible by 4')

        # Number of symmetric and non-symmetric filters
        self.filters_sym = self.symmetry['h'] + self.symmetry['v'] + self.symmetry['hv']
        self.filters_nonsym = self.filters - self.filters_sym
        if self.filters_nonsym < 0:
            raise ValueError('Number of symmetric filters exceeds total number of filters')

    def flip_filter(self, kernel, axis):
        '''Creates symmetric filters by flipping a filter along one or more axes.
           Since the tf.image functions are used to do the flipping the
           dimensions of the filters need to be permuted accordingly.
        '''
        if type(axis) is not list:
            axis = [axis]
        # image: 4-D Tensor of shape [batch, height, width, channels]
        #     or 3-D Tensor of shape [height, width, channels]
        # However, the filter tensor is [filter_height, filter_width, n_input_chans, n_filters]
        # so we first transpose it into [n_filters, filter_height, filter_width, n_input_chans]
        # treating n_filters as if it was the batch
        kernel = tf.transpose(kernel, perm=[3, 0, 1, 2])
        if 'v' in axis:
            kernel = tf.image.flip_up_down(kernel)
        if 'h' in axis:
            kernel = tf.image.flip_left_right(kernel)
        # back to original shape
        return tf.transpose(kernel, perm=[1, 2, 3, 0])

    def build(self, input_shape):
        # based on class _Conv
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        #input_shape = input_shape.as_list()
        input_dim = int(input_shape[channel_axis])

        # --- Start symmetric kernel definition ---
        # Specify first 3 dimensions of kernel shape
        shape = self.kernel_size + (input_dim,)
        kernels = []

        # Create symmetric filter pairs: symmetric about y (vertical) axis
        if self.symmetry['h'] > 0:
            x = self.add_weight(name='kernel_sym_h',
                                          shape=shape + (self.symmetry['h']//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='h')])

        # Create symmetric filter pairs: symmetric about x (horizontal) axis
        if self.symmetry['v'] > 0:
            x = self.add_weight(name='kernel_sym_v',
                                          shape=shape + (self.symmetry['v']//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='v')])

        # Create symmetric filter quadruples: symmetric about x or y axes
        if self.symmetry['hv'] > 0:
            x = self.add_weight(name='kernel_sym_hv',
                                          shape=shape + (self.symmetry['hv']//4,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='h'), \
                               self.flip_filter(x, axis='v'), \
                               self.flip_filter(x, axis=['h', 'v'])])


        # Expand dims of symmetric kernels from 3D to 4D
        #kernels = [tf.expand_dims(x, axis=3) for x in kernels]

        # Build non-symmetric filter kernels
        if self.filters_nonsym > 0:
            kernels.append(self.add_weight(shape=shape + (self.filters_nonsym,),
                                          initializer=self.kernel_initializer,
                                          name='kernel_nonsym',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype))

        # Concatenate all kernels
        self.kernel = tf.concat(kernels, -1)

        # --- End symmetric kernel definition ---

        # Bias term
        if self.use_bias:
            if self.share_bias:
                # Make symmetric filter pairs share the bias term as well
                biases = []
                if self.symmetry['v'] > 0:
                    x = self.add_weight(shape=(self.symmetry['v']//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_v',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.symmetry['h'] > 0:
                    x = self.add_weight(shape=(self.symmetry['h']//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_h',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.symmetry['hv'] > 0:
                    x = self.add_weight(shape=(self.symmetry['hv']//4,),
                                                initializer=self.bias_initializer,
                                                name='bias_hv',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x, x, x])
                if self.filters_nonsym > 0:
                    x = self.add_weight(shape=(self.filters_nonsym,),
                                                initializer=self.bias_initializer,
                                                name='bias_nonsym',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.append(x)
                self.bias = tf.concat(biases, -1)

            else:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        else:
            self.bias = None

        # Set input spec.
        self.input_spec = layers.InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})

        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()

        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.get_shape(),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding,
            data_format=conv_utils.convert_data_format(self.data_format,
    self.rank + 2))
        self.built = True

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
