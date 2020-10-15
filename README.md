# CNN-Symmetry
Implementation of reflection symmetry in convolutional layers (Pytorch and Tensorflow 1)

Symmetry is implemented via weight sharing *between* filter kernels. Example: Consider a 3 x 3 convolutional kernel with the coefficients

```
0.5,  0.1, -0.3
0.0 , 1.2, -0.9
0.0,  0.0,  0.1
```

It's horizontal reflection (coded as `'h'`) is given by 


```
-0.3,  0.1, 0.5
-0.9 , 1.2, 0.0
 0.1,  0.0, 0.0
```


and its vertical reflection version (coded as `'v'`) is given by 


```
0.0,  0.0,  0.1
0.0 , 1.2, -0.9
0.5,  0.1, -0.3
```

We then say these filters come in a (horizontally or vertically symmetrc) pair. Only 3x3=9 weights are then needed to represent two filters instead of 2x3x3=18 weights. We can consider combinations of both vertical and horizontal symmetry (coded as `'hv'`) that gives us a filter quadruple: original filter, horizontal reflection, vertical reflection, horizontal+vertical reflection, so only 25% of the weights are required.  For 3-D convolutions, we can extend the same principle to the depth axis (coded as `'z'`).

## Pytorch

The Python file `custom_layers_torch.py` provides symmetric convolutional layers for 2-D and 3-D convolutions. Both layers extend the Pytorch layers `torch.nn.Conv2d` and `torch.nn.Conv3d` and can be used as drop-in replacements for these layers. Consider the following simple CNN with three convolutional layers:

```python
class CNN_2D(nn.Module):

    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, kernel_size=3)
        self.batchnorm = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 32, kernel_size=3)
        self.conv_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(10*10*self.conv3.out_channels, 1024)
        self.fc_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(self.fc1.out_features, 1)

    def forward(self, x):
        ...
```

Assume we want to replace the first convolutional layer with a symmetric layer.

```python
from symmetric_layers_torch import SymmetricConv2d, SymmetricConv3d

class CNN_2D(nn.Module):

    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = SymmetricConv2d(in_channels=1, out_channels=16, kernel_size=3, \
            symmetry={'h':2, 'v':4, 'hv':8})
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, kernel_size=3)
        self.batchnorm = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 32, kernel_size=3)
        self.conv_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(10*10*self.conv3.out_channels, 1024)
        self.fc_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(self.fc1.out_features, 1)

    def forward(self, x):
        ...
```

Symmetry layers take an additional parameter `symmetry`. In the example `symmetry = {'h':2, 'v':4, 'hv':8}` it is specified that 2 filters are to be horizontally symmetric (giving 1 filter pair), 4 vertically symmetric (giving 2 filter pairs), and 8 horizontally/vertically symmetric (giving 2 filter quadruples). They specify 2+4+8=14 filters which is less than the total number of filters (`out_channels=16`). In this case, 2 additional filters get added that are ordinary, non-symmetric convolutional filters.
The extension to 3-D convolutions is seamless, just provide an additional key `'z'` which encodes the 'depth' dimension e.g. `symmetry={'h':2, 'v':2, 'z':2, 'hz':4}`.


## Tensorflow

The Python file `custom_layers_tf1.py` provides a symmetric layer (and other custom layers) for Tensorflow 1.X:

- `SymmetricConv2D`: extension of `Conv2D` that adds weight sharing between pairs of filters (horizontal or vertical reflection symmetry). Acts exactly like `Conv2D`, but it has an additional argument `symmetry` that accepts a dictionary specifying the amount of symmetric filter pairs. E.g. `{'h': 16, 'v': 8, 'hv': 8}` specifies 16 horizontally symmetric filters, 8 vertically symmetric, and 8 filter quadruples that are horizontally and/or vertically symmetric. Alternatively, percentages can be given as strings eg `{'h':'50%', 'v':'50%'}`. Note that for `h` and `v` filters the number of weights is reduced to 50\%, whereas it is reduced to 25\% for `hv` filters.
- `GlobalConv1D`: 1D convolution that is invariant wrt permutations of the features. Every convolutional kernel has two, one for the center (the value) the kernel is centered on, and one for the surround (all other values)
- `TopKPool`: selects the top-k values (equal to Maxpooling for k=1)


Snippet using `SymmetricConv2D` with Tensorflow/Keras.

```
from custom_layers import SymmetricConv2D

symmetry = {'h':'50%', 'v':'50%'}

inputs = layers.Input(im.shape[1:])
x = SymmetricConv2D(32, (5, 5), symmetry=sym, share_bias=share_bias)(inputs)

# ... more layers here ...
```
