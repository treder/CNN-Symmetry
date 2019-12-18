# CNN-Symmetry
Implementation of reflection symmetry in convolutional layers (Tensorflow 1.X)


The Python file `custom_layers.py` provides a symmetric layer (and other custom layers) for Tensorflow 1.X:

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
