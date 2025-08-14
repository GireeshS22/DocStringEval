class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    .. versionadded:: 0.20

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

        .. versionadded:: 0.24

    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.

        .. versionadded:: 0.24

    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.

        .. versionadded:: 1.1

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.3
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    max_categories : int, default=None
        Specifies an upper limit to the number of output categories for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        `max_categories` do **not** take into account missing or unknown
        categories. Setting `unknown_value` or `encoded_missing_value` to an
        integer will increase the number of unique integer codes by one each.
        This can result in up to `max_categories + 2` integer codes.

        .. versionadded:: 1.3
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.3

    See Also
    --------
    OneHotEncoder : Performs a one-hot encoding of categorical features. This encoding
        is suitable for low to medium cardinality categorical variables, both in
        supervised and unsupervised settings.
    TargetEncoder : Encodes categorical features using supervised signal
        in a classification or regression pipeline. This encoding is typically
        suitable for high cardinality categorical variables.
    LabelEncoder : Encodes target labels with values between 0 and
        ``n_classes-1``.

    Notes
    -----
    With a high proportion of `nan` values, inferring categories becomes slow with
    Python versions before 3.10. The handling of `nan` values was improved
    from Python 3.10 onwards, (c.f.
    `bpo-43475 <https://github.com/python/cpython/issues/87641>`_).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    By default, :class:`OrdinalEncoder` is lenient towards missing values by
    propagating them.

    >>> import numpy as np
    >>> X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
    >>> enc.fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., nan]])

    You can use the parameter `encoded_missing_value` to encode missing values.

    >>> enc.set_params(encoded_missing_value=-1).fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., -1.]])

    Infrequent categories are enabled by setting `max_categories` or `min_frequency`.
    In the following example, "a" and "d" are considered infrequent and grouped
    together into a single category, "b" and "c" are their own categories, unknown
    values are encoded as 3 and missing values are encoded as 4.

    >>> X_train = np.array(
    ...     [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]],
    ...     dtype=object).T
    >>> enc = OrdinalEncoder(
    ...     handle_unknown="use_encoded_value", unknown_value=3,
    ...     max_categories=3, encoded_missing_value=4)
    >>> _ = enc.fit(X_train)
    >>> X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"], [np.nan]], dtype=object)
    >>> enc.transform(X_test)
    array([[2.],
           [0.],
           [1.],
           [2.],
           [3.],
           [4.]])
    """
    _parameter_constraints: dict = {'categories': [StrOptions({'auto'}), list], 'dtype': 'no_validation', 'encoded_missing_value': [Integral, type(np.nan)], 'handle_unknown': [StrOptions({'error', 'use_encoded_value'})], 'unknown_value': [Integral, type(np.nan), None], 'max_categories': [Interval(Integral, 1, None, closed='left'), None], 'min_frequency': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='neither'), None]}

    def __init__(self, *, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        if self.handle_unknown == 'use_encoded_value':
            if is_scalar_nan(self.unknown_value):
                if np.dtype(self.dtype).kind != 'f':
                    raise ValueError(f'When unknown_value is np.nan, the dtype parameter should be a float dtype. Got {self.dtype}.')
            elif not isinstance(self.unknown_value, numbers.Integral):
                raise TypeError(f"unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value', got {self.unknown_value}.")
        elif self.unknown_value is not None:
            raise TypeError(f"unknown_value should only be set when handle_unknown is 'use_encoded_value', got {self.unknown_value}.")
        fit_results = self._fit(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', return_and_ignore_missing_for_infrequent=True)
        self._missing_indices = fit_results['missing_indices']
        cardinalities = [len(categories) for categories in self.categories_]
        if self._infrequent_enabled:
            for feature_idx, infrequent in enumerate(self.infrequent_categories_):
                if infrequent is not None:
                    cardinalities[feature_idx] -= len(infrequent)
        for cat_idx, categories_for_idx in enumerate(self.categories_):
            if is_scalar_nan(categories_for_idx[-1]):
                cardinalities[cat_idx] -= 1
        if self.handle_unknown == 'use_encoded_value':
            for cardinality in cardinalities:
                if 0 <= self.unknown_value < cardinality:
                    raise ValueError(f'The used value for unknown_value {self.unknown_value} is one of the values already used for encoding the seen categories.')
        if self._missing_indices:
            if np.dtype(self.dtype).kind != 'f' and is_scalar_nan(self.encoded_missing_value):
                raise ValueError(f'There are missing values in features {list(self._missing_indices)}. For OrdinalEncoder to encode missing values with dtype: {self.dtype}, set encoded_missing_value to a non-nan value, or set dtype to a float')
            if not is_scalar_nan(self.encoded_missing_value):
                invalid_features = [cat_idx for cat_idx, cardinality in enumerate(cardinalities) if cat_idx in self._missing_indices and 0 <= self.encoded_missing_value < cardinality]
                if invalid_features:
                    if hasattr(self, 'feature_names_in_'):
                        invalid_features = self.feature_names_in_[invalid_features]
                    raise ValueError(f'encoded_missing_value ({self.encoded_missing_value}) is already used to encode a known category in features: {invalid_features}')
        return self

    def transform(self, X):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        check_is_fitted(self, 'categories_')
        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', ignore_category_indices=self._missing_indices)
        X_trans = X_int.astype(self.dtype, copy=False)
        for cat_idx, missing_idx in self._missing_indices.items():
            X_missing_mask = X_int[:, cat_idx] == missing_idx
            X_trans[X_missing_mask, cat_idx] = self.encoded_missing_value
        if self.handle_unknown == 'use_encoded_value':
            X_trans[~X_mask] = self.unknown_value
        return X_trans

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, force_all_finite='allow-nan')
        n_samples, _ = X.shape
        n_features = len(self.categories_)
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        dt = np.result_type(*[cat.dtype for cat in self.categories_])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        found_unknown = {}
        infrequent_masks = {}
        infrequent_indices = getattr(self, '_infrequent_indices', None)
        for i in range(n_features):
            labels = X[:, i]
            if i in self._missing_indices:
                X_i_mask = _get_mask(labels, self.encoded_missing_value)
                labels[X_i_mask] = self._missing_indices[i]
            rows_to_update = slice(None)
            categories = self.categories_[i]
            if infrequent_indices is not None and infrequent_indices[i] is not None:
                infrequent_encoding_value = len(categories) - len(infrequent_indices[i])
                infrequent_masks[i] = labels == infrequent_encoding_value
                rows_to_update = ~infrequent_masks[i]
                frequent_categories_mask = np.ones_like(categories, dtype=bool)
                frequent_categories_mask[infrequent_indices[i]] = False
                categories = categories[frequent_categories_mask]
            if self.handle_unknown == 'use_encoded_value':
                unknown_labels = _get_mask(labels, self.unknown_value)
                found_unknown[i] = unknown_labels
                known_labels = ~unknown_labels
                if isinstance(rows_to_update, np.ndarray):
                    rows_to_update &= known_labels
                else:
                    rows_to_update = known_labels
            labels_int = labels[rows_to_update].astype('int64', copy=False)
            X_tr[rows_to_update, i] = categories[labels_int]
        if found_unknown or infrequent_masks:
            X_tr = X_tr.astype(object, copy=False)
        if found_unknown:
            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None
        if infrequent_masks:
            for idx, mask in infrequent_masks.items():
                X_tr[mask, idx] = 'infrequent_sklearn'
        return X_tr