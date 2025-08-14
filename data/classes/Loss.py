@keras_export(['keras.Loss', 'keras.losses.Loss'])
class Loss(KerasSaveable):
    """Loss base class.

    This is the class to subclass in order to create new custom losses.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation using `y_true`,
        `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            return ops.mean(ops.square(y_pred - y_true), axis=-1)
    ```
    """

    def __init__(self, name=None, reduction='sum_over_batch_size', dtype=None):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)
        self._dtype_policy = dtype_policies.get(dtype or backend.floatx())
        self._dtype = self._dtype_policy.compute_dtype

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, y_true, y_pred, sample_weight=None):
        in_mask = backend.get_keras_mask(y_pred)
        with ops.name_scope(self.name):
            y_pred = tree.map_structure(lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred)
            y_true = tree.map_structure(lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true)
            losses = self.call(y_true, y_pred)
            out_mask = backend.get_keras_mask(losses)
            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None
            return reduce_weighted_values(losses, sample_weight=sample_weight, mask=mask, reduction=self.reduction, dtype=self.dtype)

    def call(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        return {'name': self.name, 'reduction': self.reduction}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _obj_type(self):
        return 'Loss'