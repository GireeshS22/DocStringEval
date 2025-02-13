class LabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). `LabelBinarizer` makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. `LabelBinarizer` makes this easy
    with the :meth:`inverse_transform` method.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Parameters
    ----------
    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False
        True if the returned array from transform is desired to be in sparse
        CSR format.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    y_type_ : str
        Represents the type of the target data as evaluated by
        :func:`~sklearn.utils.multiclass.type_of_target`. Possible type are
        'continuous', 'continuous-multioutput', 'binary', 'multiclass',
        'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

    sparse_input_ : bool
        `True` if the input data to transform is given as a sparse matrix,
         `False` otherwise.

    See Also
    --------
    label_binarize : Function to perform the transform operation of
        LabelBinarizer with fixed classes.
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.

    Examples
    --------
    >>> from sklearn.preprocessing import LabelBinarizer
    >>> lb = LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    Binary targets transform to a column vector

    >>> lb = LabelBinarizer()
    >>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    Passing a 2D matrix for multilabel classification

    >>> import numpy as np
    >>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
    LabelBinarizer()
    >>> lb.classes_
    array([0, 1, 2])
    >>> lb.transform([0, 1, 2, 1])
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]])
    """
    _parameter_constraints: dict = {'neg_label': [Integral], 'pos_label': [Integral], 'sparse_output': ['boolean']}

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y):
        """Fit label binarizer.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.neg_label >= self.pos_label:
            raise ValueError(f'neg_label={self.neg_label} must be strictly less than pos_label={self.pos_label}.')
        if self.sparse_output and (self.pos_label == 0 or self.neg_label != 0):
            raise ValueError(f'Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label={self.pos_label} and neg_label={self.neg_label}')
        self.y_type_ = type_of_target(y, input_name='y')
        if 'multioutput' in self.y_type_:
            raise ValueError('Multioutput target data is not supported with label binarization')
        if _num_samples(y) == 0:
            raise ValueError('y has 0 samples: %r' % y)
        self.sparse_input_ = sp.issparse(y)
        self.classes_ = unique_labels(y)
        return self

    def fit_transform(self, y):
        """Fit label binarizer/transform multi-class labels to binary labels.

        The output of transform is sometimes referred to as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {ndarray, sparse matrix} of shape (n_samples,) or                 (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform multi-class labels to binary labels.

        The output of transform is sometimes referred to by some authors as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {array, sparse matrix} of shape (n_samples,) or                 (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        check_is_fitted(self)
        y_is_multilabel = type_of_target(y).startswith('multilabel')
        if y_is_multilabel and (not self.y_type_.startswith('multilabel')):
            raise ValueError('The object was not fitted with multilabel input.')
        return label_binarize(y, classes=self.classes_, pos_label=self.pos_label, neg_label=self.neg_label, sparse_output=self.sparse_output)

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels.

        Parameters
        ----------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Target values. All sparse matrices are converted to CSR before
            inverse transformation.

        threshold : float, default=None
            Threshold used in the binary and multi-label cases.

            Use 0 when ``Y`` contains the output of :term:`decision_function`
            (classifier).
            Use 0.5 when ``Y`` contains the output of :term:`predict_proba`.

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target values. Sparse matrix will be of CSR format.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), :meth:`inverse_transform` chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's :term:`decision_function` method directly as the input
        of :meth:`inverse_transform`.
        """
        check_is_fitted(self)
        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.0
        if self.y_type_ == 'multiclass':
            y_inv = _inverse_binarize_multiclass(Y, self.classes_)
        else:
            y_inv = _inverse_binarize_thresholding(Y, self.y_type_, self.classes_, threshold)
        if self.sparse_input_:
            y_inv = sp.csr_matrix(y_inv)
        elif sp.issparse(y_inv):
            y_inv = y_inv.toarray()
        return y_inv

    def _more_tags(self):
        return {'X_types': ['1dlabels']}