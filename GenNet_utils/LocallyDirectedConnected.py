# For the article see https://www.biorxiv.org/content/10.1101/2020.06.19.159152v1
# For an explenation how to use this layer see https://github.com/ArnovanHilten/GenNet
# Locallyconnected1D is used as a basis to write the LocallyDirected layer
# ==============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""LocallyDirected1D layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import activations # layer activation functions 
from tensorflow.python.keras import backend as K # backend is computational engine executing operations and computations 
from tensorflow.python.keras import constraints # module for setting model parameter constraints during training
from tensorflow.python.keras import initializers # module for setting initial rarndom weights of layers
from tensorflow.python.keras import regularizers # module for regularization penalties for each layer
from tensorflow.python.keras.engine.base_layer import InputSpec # specified rank, dtype, and shape of every input to a layer
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils 
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.LocallyDirected1D')
class LocallyDirected1D(Layer):
    """Locally-Directed1D layer for 1D inputs.

  The `LocallyDirected1D` layer works similarly to
  the `Conv1D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each different patch
  of the input.

  Example:
  ```python
      # apply a unshared weight convolution 1d of length 3 to a sequence with
      # 10 timesteps, with 64 output filters
      model = Sequential()
      model.add(LocallyDirected1D(64, 3, input_shape=(10, 32)))
      # now model.output_shape == (None, 8, 64)
      # add a new conv1d on top
      model.add(LocallyDirected1D(32, 3))
      # now model.output_shape == (None, 6, 32)
  ```

  Arguments:
      mask: sparse matrix with shape (input, output) connectivity matrix,
            True defines connection between (in_i, out_j), should be sparse (False,0) >> True
            should be scipy sparese matrix in COO Format!
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
          specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
          specifying the stride length of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: Currently only supports `"valid"` (case-insensitive).
          `"same"` may be supported in the future.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, length, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, length)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`

  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.
  """

    def __init__(self,
                 mask,
                 filters,
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(LocallyDirected1D, self).__init__(**kwargs)
        self.filters = filters
        self.padding = conv_utils.normalize_padding(padding) # using keras.utils existing conv_utils to implement similar padding to Conv1D
        self.data_format = conv_utils.normalize_data_format(data_format) # using keras.utils existing conv_utils to implement similar data output formatting to Conv1D
        self.activation = activations.get(activation) # using keras activations to set attribute to existing activation function
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.mask = mask

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        """
        Build the Kernel, Kernel Mask, and Bias variables based on the passed in arguments and class attributes/shapes
        
        Input:
        - input_shape (Tuple): elements of 3 dimensions: (batch, length, channels) or (batch, channels, length) dpeneding on channels_first boolean

        Output:
        None
        """

        # Extract the channels and length dimensions of the input
        if self.data_format == 'channels_first':
            input_dim, input_length = input_shape[1], input_shape[2]
        else:
            input_dim, input_length = input_shape[2], input_shape[1]

        # Raise Error if no channels defined 
        if input_dim is None:
            raise ValueError('Axis 2 of input should be fully-defined. '
                             'Found shape:', input_shape)
        self.output_length = self.mask.shape[1]
#         print("output length is " + str(self.output_length))

        # Set Kernel Shape based on input dimensions and args
        if self.data_format == 'channels_first':
            self.kernel_shape = (input_dim, input_length,
                                 self.filters, self.output_length)
        else:
            self.kernel_shape = (input_length, input_dim,
                                 self.output_length, self.filters)

        # Set Kernel to be the new variable added to the layer with the specified kernel attributes and shape
        self.kernel = self.add_weight(shape=(len(self.mask.data),),  # sum of all nonzero values in mask sum(sum(mask))
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set Kernel mask to be the output of the get_locallyDirected1D_mask function below, see documentation for what it returns 
        self.kernel_mask = get_locallyDirected1D_mask(self.mask, self.kernel,
                                                      data_format=self.data_format,
                                                      dtype=self.kernel.dtype
                                                      )

        # Set self.bias to be new variable added to the layer with the specified bias attributes and shape
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_length, self.filters),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        # Sets attribute to specify rank, dtype, and shape of every input to layer depending on channels_first
        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=3, axes={1: input_dim})
        else:
            self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        # output = local_conv_matmul(inputs, self.kernel_mask,
        #                            self.output_length)

        """
        Perform the local N-D convolution with un-shared weights and added bias, and return the output of the activation function applied to the convolution result
        
        Input:
        - inputs: (N+2)-D tensor with shape
          `(batch_size, channels_in, d_in1, ..., d_inN)`
          or
          `(batch_size, d_in1, ..., d_inN, channels_in)`.

        Output:
        - output: Output (N+2)-D tensor with shape `output_shape`.
        """
        
        output = local_conv_matmul_sparse(inputs, self.kernel_mask,
                                          self.output_length, self.filters)

        # Add the bias to the convolution output tensor result if needed
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        # Pass the convolution output through the activation function to be returned 
        output = self.activation(output)
        return output

    def get_config(self):  # delete this?
        """
        Return a dictionary of the base_configuration super attributes as well as the attributes of this class. 
        """
        config = {
            'filters':
                self.filters,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
        }
        base_config = super(LocallyDirected1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_locallyDirected1D_mask(mask, kernel, data_format,
                               dtype):
    """Return a mask representing connectivity of a locally-connected operation.

  This method returns a masking tensor of 0s and 1s (of type `dtype`) that,
  when element-wise multiplied with a fully-connected weight tensor, masks out
  the weights between disconnected input-output pairs and thus implements local
  connectivity through a sparse fully-connected weight tensor.

  Assume an unshared convolution with given parameters is applied to an input
  having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
  to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
  by layer parameters such as `strides`).

  This method returns a mask which can be broadcast-multiplied (element-wise)
  with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer between
  (N+1)-D activations (N spatial + 1 channel dimensions for input and output)
  to make it perform an unshared convolution with given `kernel_shape`,
  `strides`, `padding` and `data_format`.

  Arguments:
    mask: sparse connectivity matrix matrix
    kernel: weights with len(sum(non-sparse values)
    data_format: a string, `"channels_first"` or `"channels_last"`.
    dtype: type of the layer operation, e.g. `tf.float64`.

  Returns:
    a `dtype`-tensor of shape
    `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
    if `data_format == `"channels_first"`, or
    `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
    if `data_format == "channels_last"`.

    Adaption by arno is now a sparse matrix.
  Raises:
    ValueError: if `data_format` is neither `"channels_first"` nor
                `"channels_last"`.
  """

    ndims = int(mask.ndim / 2)
    indices = np.mat([mask.row, mask.col]).transpose()
#     print(mask.shape)

    # Generate a sparse tensor for the mask using TF module with specified characteristics                                
    mask = tf.SparseTensor(indices, kernel, [mask.shape[0], mask.shape[1]])

    if data_format == 'channels_first':
        # add an outer axis of length 1
        mask = tf.sparse.expand_dims(mask, 0)

        # align axis for braodcasting
        mask = tf.sparse.expand_dims(mask, - ndims - 1)

    # align axes in similar manner but with flipped axis arguments
    elif data_format == 'channels_last':
        mask = tf.sparse.expand_dims(mask, ndims)
        mask = tf.sparse.expand_dims(mask, -1)

    else:
        raise ValueError('Unrecognized data_format: ' + str(data_format))

    return mask


def local_conv_matmul_sparse(inputs, kernel_mask, output_length, filters):
    """Apply N-D convolution with un-shared weights using a single matmul call.

  This method outputs `inputs . (kernel * kernel_mask)`
  (with `.` standing for matrix-multiply and `*` for element-wise multiply)
  and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
  hence perform the same operation as a convolution with un-shared
  (the remaining entries in `kernel`) weights. It also does the necessary
  reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

  Arguments:
      inputs: (N+2)-D tensor with shape
          `(batch_size, channels_in, d_in1, ..., d_inN)`
          or
          `(batch_size, d_in1, ..., d_inN, channels_in)`.
      kernel: the unshared weights for N-D convolution,
          an (N+2)-D tensor of shape:
          `(d_in1, ..., d_inN, channels_in, d_out2, ..., d_outN, channels_out)`
          or
          `(channels_in, d_in1, ..., d_inN, channels_out, d_out2, ..., d_outN)`,
          with the ordering of channels and spatial dimensions matching
          that of the input.
          Each entry is the weight between a particular input and
          output location, similarly to a fully-connected weight matrix.
      kernel_mask: a float 0/1 mask tensor of shape:
           `(d_in1, ..., d_inN, 1, d_out2, ..., d_outN, 1)`
           or
           `(1, d_in1, ..., d_inN, 1, d_out2, ..., d_outN)`,
           with the ordering of singleton and spatial dimensions
           matching that of the input.
           Mask represents the connectivity pattern of the layer and is
           precomputed elsewhere based on layer parameters: stride,
           padding, and the receptive field shape.
      output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)`
          or
          `(batch_size, d_out1, ..., d_outN, channels_out)`,
          with the ordering of channels and spatial dimensions matching that of
          the input.

  Returns:
      Output (N+2)-D tensor with shape `output_shape`.
  """
    # Reshape the inputs so that it is flattened on a particular axis 
    inputs_flat = K.reshape(inputs, (K.shape(inputs)[0], -1))

    # Reshape the kernel mask tensor into a 2D tensor
    kernel_mask = make_2d_sparse(kernel_mask, split_dim=K.ndim(kernel_mask) // 2)

    # Matrix multiply kernel_mask and flattened inputs
    output_flat = tf.sparse.matmul(kernel_mask, inputs_flat, adjoint_a=True, adjoint_b=True)
    output_flat = tf.transpose(output_flat)

    # Reshape the output_flattened into the required shape
    output = K.reshape(output_flat, [-1, output_length, filters])

    return output


def make_2d_sparse(tensor, split_dim):
    """Reshapes an N-dimensional tensor into a 2D tensor.

  Dimensions before (excluding) and after (including) `split_dim` are grouped
  together.

  Arguments:
    tensor: a tensor of shape `(d0, ..., d(N-1))`.
    split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).

  Returns:
    Tensor of shape
    `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
  """

    shape = K.array_ops.shape(tensor)
    in_dims = shape[:split_dim]
    out_dims = shape[split_dim:]

    # multiply elements in the shape dimension across an axis to accumulate the total in and out size dimensions for reshaping
    in_size = K.math_ops.reduce_prod(in_dims)
    out_size = K.math_ops.reduce_prod(out_dims)

    return tf.sparse.reshape(tensor, (in_size, out_size))


def local_conv_matmul(inputs, kernel_mask, output_length):
    """Apply N-D convolution with un-shared weights using a single matmul call.

  This method outputs `inputs . (kernel * kernel_mask)`
  (with `.` standing for matrix-multiply and `*` for element-wise multiply)
  and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
  hence perform the same operation as a convolution with un-shared
  (the remaining entries in `kernel`) weights. It also does the necessary
  reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

  Arguments:
      inputs: (N+2)-D tensor with shape
          `(batch_size, channels_in, d_in1, ..., d_inN)`
          or
          `(batch_size, d_in1, ..., d_inN, channels_in)`.
      kernel: the unshared weights for N-D convolution,
          an (N+2)-D tensor of shape:
          `(d_in1, ..., d_inN, channels_in, d_out2, ..., d_outN, channels_out)`
          or
          `(channels_in, d_in1, ..., d_inN, channels_out, d_out2, ..., d_outN)`,
          with the ordering of channels and spatial dimensions matching
          that of the input.
          Each entry is the weight between a particular input and
          output location, similarly to a fully-connected weight matrix.
      kernel_mask: a float 0/1 mask tensor of shape:
           `(d_in1, ..., d_inN, 1, d_out2, ..., d_outN, 1)`
           or
           `(1, d_in1, ..., d_inN, 1, d_out2, ..., d_outN)`,
           with the ordering of singleton and spatial dimensions
           matching that of the input.
           Mask represents the connectivity pattern of the layer and is
           precomputed elsewhere based on layer parameters: stride,
           padding, and the receptive field shape.
      output_shape: a tuple of (N+2) elements representing the output shape:
          `(batch_size, channels_out, d_out1, ..., d_outN)`
          or
          `(batch_size, d_out1, ..., d_outN, channels_out)`,
          with the ordering of channels and spatial dimensions matching that of
          the input.

  Returns:
      Output (N+2)-D tensor with shape `output_shape`.
  """
    inputs_flat = K.reshape(inputs, (K.shape(inputs)[0], -1))

    kernel = make_2d_sparse(kernel_mask, split_dim=K.ndim(kernel_mask) // 2)

    output_flat = tf.sparse_tensor_dense_matmul(inputs_flat, kernel, b_is_sparse=True)
    output = K.reshape(output_flat, [-1, output_length, 1])
    return output


def make_2d(tensor, split_dim):
    """Reshapes an N-dimensional tensor into a 2D tensor.

  Dimensions before (excluding) and after (including) `split_dim` are grouped
  together.

  Arguments:
    tensor: a tensor of shape `(d0, ..., d(N-1))`.
    split_dim: an integer from 1 to N-1, index of the dimension to group
        dimensions before (excluding) and after (including).

  Returns:
    Tensor of shape
    `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
  """
#     print(tensor.shape)
    shape = K.array_ops.shape(tensor)
    in_dims = shape[:split_dim]
    out_dims = shape[split_dim:]

    in_size = K.math_ops.reduce_prod(in_dims)
    out_size = K.math_ops.reduce_prod(out_dims)

    return K.array_ops.reshape(tensor, (in_size, out_size))

# %%
