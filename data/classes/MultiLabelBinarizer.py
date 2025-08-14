class MultiLabelBinarizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """Transform between iterable of iterables and a multilabel format.

    Although a list of sets or tuples is a very intuitive format for multilabel
    data, it is unwieldy to process. This transformer converts between this
    intuitive format and the supported multilabel format: a (samples x classes)
    binary matrix indicating the presence of a class label.

    Parameters
    ----------
    classes : array-like of shape (n_classes,), default=None
        Indicates an ordering for the class labels.
        All entries should be unique (cannot contain duplicate classes).

    sparse_output : bool, default=False
        Set to True if output binary array is desired in CSR sparse format.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        A copy of the `classes` parameter when provided.
        Otherwise it corresponds to the sorted set of classes found
        when fitting.

    See Also
    --------
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.

    Examples
    --------
    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> mlb.classes_
    array([1, 2, 3])

    >>> mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}])
    array([[0, 1, 1],
           [1, 0, 0]])
    >>> list(mlb.classes_)
    ['comedy', 'sci-fi', 'thriller']

    A common mistake is to pass in a list, which leads to the following issue:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit(['sci-fi', 'thriller', 'comedy'])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['-', 'c', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'o', 'r', 's', 't',
        'y'], dtype=object)

    To correct this, the list of labels should be passed in as:

    >>> mlb = MultiLabelBinarizer()
    >>> mlb.fit([['sci-fi', 'thriller', 'comedy']])
    MultiLabelBinarizer()
    >>> mlb.classes_
    array(['comedy', 'sci-fi', 'thriller'], dtype=object)
    """
    _parameter_constraints: dict = {'classes': ['array-like', None], 'sparse_output': ['boolean']}

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y):
        """Fit the label sets binarizer, storing :term:`classes_`.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._cached_dict = None
        if self.classes is None:
            classes = sorted(set(itertools.chain.from_iterable(y)))
        elif len(set(self.classes)) < len(self.classes):
            raise ValueError('The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer.')
        else:
            classes = self.classes
        dtype = int if all((isinstance(c, int) for c in classes)) else object
        self.classes_ = np.empty(len(classes), dtype=dtype)
        self.classes_[:] = classes
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, y):
        """Fit the label sets binarizer and transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]`
            is in `y[i]`, and 0 otherwise. Sparse matrix will be of CSR
            format.
        """
        if self.classes is not None:
            return self.fit(y).transform(y)
        self._cached_dict = None
        class_mapping = defaultdict(int)
        class_mapping.default_factory = class_mapping.__len__
        yt = self._transform(y, class_mapping)
        tmp = sorted(class_mapping, key=class_mapping.get)
        dtype = int if all((isinstance(c, int) for c in tmp)) else object
        class_mapping = np.empty(len(tmp), dtype=dtype)
        class_mapping[:] = tmp
        self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
        yt.indices = np.asarray(inverse[yt.indices], dtype=yt.indices.dtype)
        if not self.sparse_output:
            yt = yt.toarray()
        return yt

    def transform(self, y):
        """Transform the given label sets.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        Returns
        -------
        y_indicator : array or CSR matrix, shape (n_samples, n_classes)
            A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in
            `y[i]`, and 0 otherwise.
        """
        check_is_fitted(self)
        class_to_index = self._build_cache()
        yt = self._transform(y, class_to_index)
        if not self.sparse_output:
            yt = yt.toarray()
        return yt

    def _build_cache(self):
        if self._cached_dict is None:
            self._cached_dict = dict(zip(self.classes_, range(len(self.classes_))))
        return self._cached_dict

    def _transform(self, y, class_mapping):
        """Transforms the label sets with a given mapping.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        class_mapping : Mapping
            Maps from label to column index in label indicator matrix.

        Returns
        -------
        y_indicator : sparse matrix of shape (n_samples, n_classes)
            Label indicator matrix. Will be of CSR format.
        """
        indices = array.array('i')
        indptr = array.array('i', [0])
        unknown = set()
        for labels in y:
            index = set()
            for label in labels:
                try:
                    index.add(class_mapping[label])
                except KeyError:
                    unknown.add(label)
            indices.extend(index)
            indptr.append(len(indices))
        if unknown:
            warnings.warn('unknown class(es) {0} will be ignored'.format(sorted(unknown, key=str)))
        data = np.ones(len(indices), dtype=int)
        return sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(class_mapping)))

    def inverse_transform(self, yt):
        """Transform the given indicator matrix into label sets.

        Parameters
        ----------
        yt : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            A matrix containing only 1s ands 0s.

        Returns
        -------
        y : list of tuples
            The set of labels for each sample such that `y[i]` consists of
            `classes_[j]` for each `yt[i, j] == 1`.
        """
        check_is_fitted(self)
        if yt.shape[1] != len(self.classes_):
            raise ValueError('Expected indicator for {0} classes, but got {1}'.format(len(self.classes_), yt.shape[1]))
        if sp.issparse(yt):
            yt = yt.tocsr()
            if len(yt.data) != 0 and len(np.setdiff1d(yt.data, [0, 1])) > 0:
                raise ValueError('Expected only 0s and 1s in label indicator.')
            return [tuple(self.classes_.take(yt.indices[start:end])) for start, end in zip(yt.indptr[:-1], yt.indptr[1:])]
        else:
            unexpected = np.setdiff1d(yt, [0, 1])
            if len(unexpected) > 0:
                raise ValueError('Expected only 0s and 1s in label indicator. Also got {0}'.format(unexpected))
            return [tuple(self.classes_.compress(indicators)) for indicators in yt]

    def _more_tags(self):
        return {'X_types': ['2dlabels']}