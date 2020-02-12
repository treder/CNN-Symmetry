# Symmetric layers for Tensorflow 1.X:
# - SymmetricConv2D: extension of Conv2D that adds weight sharing between pairs of
#                    filters (horizontal or vertical reflection symmetry)
# - SymmetricConv2DTranspose: symmetric extension of Conv2DTranspose

# (c) matthias treder 2020

import numpy as np
import re, math

import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.keras import models, layers, Model
from tensorflow.keras.layers import Layer, Input, Conv2DTranspose
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
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
        # make sure it's not checkpointed to not raise errors when saving weights
        sym = dict(symmetry) # make a copy
        sym.setdefault('h', 0)
        sym.setdefault('v', 0)
        sym.setdefault('hv', 0)
        self.share_bias = share_bias

        # If symmetric filters are specified as percentages, calculate the
        # absolute numbers here
        for key, val in sym.items():
            if (type(val) is str) and (val[-1]=='%'):
                val = round(float(val[:-1]) * self.filters / 100)
                # Make sure number of filters is divisible by 2 or 4
                if key in ['h','v']:
                    val = (val // 2) * 2
                elif key=='hv':
                    val = (val // 4) * 4
                sym[key] = val

        # Check if number of filters is divisible by 2 resp. 4
        for key, val in sym.items():
                if (key in ['h','v']) and (val % 2 != 0):
                    raise ValueError('Number of symmetric h and v filters must be divisible by 2')
                elif (key=='hv') and (val % 4 != 0):
                    raise ValueError('Number of symmetric hv filters must be divisible by 4')

        # Number of symmetric and non-symmetric filters
        self.filters_sym = sym['h'] + sym['v'] + sym['hv']
        self.filters_nonsym = self.filters - self.filters_sym
        if self.filters_nonsym < 0:
            raise ValueError('Number of symmetric filters exceeds total number of filters')

        self.h = sym['h']
        self.v = sym['v']
        self.hv = sym['hv']
        print('Symmetry:', self.h, self.v, self.hv)


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
        if self.h > 0:
            x = self.add_weight(name='kernel_sym_h',
                                          shape=shape + (self.h//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='h')])

        # Create symmetric filter pairs: symmetric about x (horizontal) axis
        if self.v > 0:
            x = self.add_weight(name='kernel_sym_v',
                                          shape=shape + (self.v//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='v')])

        # Create symmetric filter quadruples: symmetric about x or y axes
        if self.hv > 0:
            x = self.add_weight(name='kernel_sym_hv',
                                          shape=shape + (self.hv//4,),
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
                if self.v > 0:
                    x = self.add_weight(shape=(self.v//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_v',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.h > 0:
                    x = self.add_weight(shape=(self.h//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_h',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.hv > 0:
                    x = self.add_weight(shape=(self.hv//4,),
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

class SymmetricConv2DTranspose(SymmetricConv2D):

    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               output_padding=None,
               dilation_rate=(1, 1),
               **kwargs):

        super(SymmetricConv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                               'greater than output padding ' +
                               str(self.output_padding))
    def build(self, input_shape):
        # like in SymmetricConv2D but the last part on op_padding
        # and convolution_op has been removed
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
        if self.h > 0:
            x = self.add_weight(name='kernel_sym_h',
                                          shape=shape + (self.h//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='h')])

        # Create symmetric filter pairs: symmetric about x (horizontal) axis
        if self.v > 0:
            x = self.add_weight(name='kernel_sym_v',
                                          shape=shape + (self.v//2,),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
            kernels.extend([x, self.flip_filter(x, axis='v')])

        # Create symmetric filter quadruples: symmetric about x or y axes
        if self.hv > 0:
            x = self.add_weight(name='kernel_sym_hv',
                                          shape=shape + (self.hv//4,),
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
                if self.v > 0:
                    x = self.add_weight(shape=(self.v//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_v',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.h > 0:
                    x = self.add_weight(shape=(self.h//2,),
                                                initializer=self.bias_initializer,
                                                name='bias_h',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)
                    biases.extend([x, x])
                if self.hv > 0:
                    x = self.add_weight(shape=(self.hv//4,),
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

        self.built = True

    def call(self, inputs):
        # following code based on Conv2DTranspose in TF 1.8.0
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     # output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    # output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        out_shape = self.compute_output_shape(inputs.shape)
        outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = super(SymmetricConv2DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config

