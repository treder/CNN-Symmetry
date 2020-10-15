'''Symmetric convolution layers for PyTorch

This module provides two layers that can be used in 2D Convolutional Neural Networks. 
SymmetricConv2D provides an extension of Conv2D that adds weight sharing between pairs of
filters (horizontal or vertical reflection symmetry). SymmetricConv2DTranspose provides a symmetric 
extension of Conv2DTranspose.

List of classes:
- SymmetricConv2D
- SymmetricConv2DTranspose

'''
# (c) matthias treder

import torch, math
import torch.nn as nn
from torch.nn.parameter import Parameter

class SymmetricConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            symmetry: dict = {},
            share_bias: bool = False
    ):     
        '''
        Args:
        symmetry (dict) - number of filters that are symmetric about the horizontal, 
                          vertical, or both axes
                          e.g. {'h':4, 'v': 2, 'hv':8} has 4 filters (2 filter pairs) that are 
                          horizontally symmetric, 2 filters (1 filter pair) which are vertically 
                          symmetric, and 8 filters (2 filter quadruples) that are symmetric 
                          about both axes
        share_bias (bool) - if True, symmetric filter pairs also share their biases
        '''   
        super(SymmetricConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        if self.groups > 1: raise ValueError(self.__str__() + ' does not support groups>1')
        if not bias: 
            self.share_bias = False
        else:
            self.share_bias = share_bias
        if symmetry is None: 
            # no symmetry, return a standard Conv2d
            self.symmetry = None
        else:
            # Set defaults for symmetric filters pairs
            symmetry = dict(symmetry) # make a copy
            symmetry.setdefault('h', 0)
            symmetry.setdefault('v', 0)
            symmetry.setdefault('hv', 0)
            self.symmetry = symmetry

            # sanity check: number of filters divisible by 2 resp. 4?
            for key, val in symmetry.items():
                    if (key in ['h','v']) and (val % 2 != 0):
                        raise ValueError('Number of symmetric h and v filters must be divisible by 2')
                    elif (key=='hv') and (val % 4 != 0):
                        raise ValueError('Number of symmetric hv filters must be divisible by 4')
            # sanity check: number of symmetric filters must be <= number of filters
            assert sum(list(symmetry.values())) <= self.out_channels, "Number of symmetric channels exceeds number of out channels"
            self.unique_out_channels = self.out_channels - symmetry['h']//2 - symmetry['v']//2 - 3*symmetry['hv']//4

            # Create only the unique weights 
            if self.transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, self.unique_out_channels, *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    self.unique_out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def forward(self, input):
        '''
        Starting from the unique weights, use torch.flip calls to create their 
        symmetric counterparts. Then concatenate all kernels and forward the resultant weights.
        '''
        s = self.symmetry
        weight = [self.weight]
        ix = 0
        if s['h'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['h']//2,:,:,:], (3,)))
            ix += s['h']//2
        if s['v'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['v']//2,:,:,:], (2,)))
            ix += s['v']//2
        if s['hv'] > 0:
            n = s['hv']//4
            weight.extend([torch.flip(self.weight[ix:ix + n,:,:,:], (3,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,3))])
            ix += n

        return self.conv2d_forward(input, torch.cat(weight, dim=0))

class SymmetricConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros', 
            symmetry: dict = {},
            share_bias: bool = False
    ):     
        '''
        Args:
        symmetry (dict) - number of filters that are symmetric about the horizontal, 
                          vertical, z axis, or any combination of them
                          e.g. {'h':4, 'z': 2, 'hv':8, 'hvz':8} has 4 filters (2 filter pairs) that are 
                          horizontally symmetric, 2 filters (1 filter pair) which are symmetric 
                          about the z axis, 8 filters (2 filter quadruples) that are symmetric 
                          horizontally and vertically, and 8 filters (1 set of 8 filters) that are
                          symmetric about all three axes
        share_bias (bool) - if True, symmetric filter pairs also share their biases
        '''   
        super(SymmetricConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        if self.groups > 1: raise ValueError(self.__str__() + ' does not support groups>1')
        if not bias: 
            self.share_bias = False
        else:
            self.share_bias = share_bias
        if symmetry is None: 
            # no symmetry, return a standard Conv2d
            self.symmetry = None
        else:
            # Set defaults for symmetric filters pairs
            symmetry = dict(symmetry) # make a copy
            symmetry.setdefault('h', 0)
            symmetry.setdefault('v', 0)
            symmetry.setdefault('z', 0)
            symmetry.setdefault('hv', 0)
            symmetry.setdefault('hz', 0)
            symmetry.setdefault('vz', 0)
            symmetry.setdefault('hvz', 0)
            self.symmetry = symmetry

            # sanity check: number of filters divisible by 2 resp. 4?
            for key, val in symmetry.items():
                    if key not in ('h','v','z','hv','hz','vz','hvz'):
                        raise ValueError("Unknown key, use only 'h','v','z','hv','hz','vz','hvz'")
                    elif (key in ('h','v','z')) and (val % 2 != 0):
                        raise ValueError('Number of symmetric h and v filters must be divisible by 2')
                    elif (key in ('hv','hz','vz')) and (val % 4 != 0):
                        raise ValueError('Number of symmetric hv filters must be divisible by 4')
                    elif (key=='hvz') and (val % 8 != 0):
                        raise ValueError('Number of symmetric hv filters must be divisible by 8')
            # sanity check: number of symmetric filters must be <= number of filters
            assert sum(list(symmetry.values())) <= self.out_channels, "Number of symmetric channels exceeds number of out channels"
            self.unique_out_channels = self.out_channels - symmetry['h']//2 - symmetry['v']//2 - symmetry['z']//2 \
                - 3*symmetry['hv']//4 - 3*symmetry['hz']//4 - 3*symmetry['vz']//4 \
                - 7*symmetry['hvz']//8


            # Create only the unique weights 
            if self.transposed:
                self.weight = Parameter(torch.Tensor(
                    in_channels, self.unique_out_channels, *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(
                    self.unique_out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def forward(self, input):
        '''
        Starting from the unique weights, use torch.flip calls to create their 
        symmetric counterparts. Then concatenate all kernels and forward the resultant weights.
        '''
        s = self.symmetry
        weight = [self.weight]
        ix = 0
        if s['h'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['h']//2,:,:,:], (4,)))
            ix += s['h']//2
        if s['v'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['v']//2,:,:,:], (3,)))
            ix += s['v']//2
        if s['z'] > 0:
            weight.append(torch.flip(self.weight[ix:ix+s['z']//2,:,:,:], (2,)))
            ix += s['z']//2
        if s['hv'] > 0:
            n = s['hv']//4
            weight.extend([torch.flip(self.weight[ix:ix + n,:,:,:], (4,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (3,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (3,4))])
            ix += n
        if s['hz'] > 0:
            n = s['hz']//4
            weight.extend([torch.flip(self.weight[ix:ix + n,:,:,:], (4,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,4))])
            ix += n
        if s['vz'] > 0:
            n = s['vz']//4
            weight.extend([torch.flip(self.weight[ix:ix + n,:,:,:], (3,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,3))])
            ix += n
        if s['hvz'] > 0:
            n = s['hvz']//8
            weight.extend([
            torch.flip(self.weight[ix:ix + n,:,:,:], (4,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (3,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,3)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,4)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (3,4)),
            torch.flip(self.weight[ix:ix + n,:,:,:], (2,3,4))
            ])
            ix += n
        return self.conv3d_forward(input, torch.cat(weight, dim=0))
