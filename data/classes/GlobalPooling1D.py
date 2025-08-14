class GlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers."""

    def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
        super(GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.keepdims = keepdims

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if self.keepdims:
                return tensor_shape.TensorShape([input_shape[0], input_shape[1], 1])
            else:
                return tensor_shape.TensorShape([input_shape[0], input_shape[1]])
        elif self.keepdims:
            return tensor_shape.TensorShape([input_shape[0], 1, input_shape[2]])
        else:
            return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format, 'keepdims': self.keepdims}
        base_config = super(GlobalPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))