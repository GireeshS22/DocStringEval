class GlobalPooling3D(Layer):
    """Abstract class for different global pooling 3D layers."""

    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super(GlobalPooling3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)
        self.keepdims = keepdims

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            if self.keepdims:
                return tensor_shape.TensorShape([input_shape[0], 1, 1, 1, input_shape[4]])
            else:
                return tensor_shape.TensorShape([input_shape[0], input_shape[4]])
        elif self.keepdims:
            return tensor_shape.TensorShape([input_shape[0], input_shape[1], 1, 1, 1])
        else:
            return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format, 'keepdims': self.keepdims}
        base_config = super(GlobalPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))