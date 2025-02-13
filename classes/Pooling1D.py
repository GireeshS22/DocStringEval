class Pooling1D(Layer):
    """Pooling layer for arbitrary pooling functions, for 1D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Args:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, steps, features)` while `channels_first`
      corresponds to inputs with shape
      `(batch, features, steps)`.
    name: A string, the name of the layer.
  """

    def __init__(self, pool_function, pool_size, strides, padding='valid', data_format='channels_last', name=None, **kwargs):
        super(Pooling1D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = backend.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_function = pool_function
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=3)

    def call(self, inputs):
        pad_axis = 2 if self.data_format == 'channels_last' else 3
        inputs = array_ops.expand_dims(inputs, pad_axis)
        outputs = self.pool_function(inputs, self.pool_size + (1,), strides=self.strides + (1,), padding=self.padding, data_format=self.data_format)
        return array_ops.squeeze(outputs, pad_axis)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps, self.pool_size[0], self.padding, self.strides[0])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape([input_shape[0], features, length])
        else:
            return tensor_shape.TensorShape([input_shape[0], length, features])

    def get_config(self):
        config = {'strides': self.strides, 'pool_size': self.pool_size, 'padding': self.padding, 'data_format': self.data_format}
        base_config = super(Pooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))