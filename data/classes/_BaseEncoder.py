class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, force_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
        if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            needs_validation = force_all_finite
        n_samples, n_features = X.shape
        X_columns = []
        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation)
            X_columns.append(Xi)
        return (X_columns, n_samples, n_features)

    def _fit(self, X, handle_unknown='error', force_all_finite=True, return_counts=False, return_and_ignore_missing_for_infrequent=False):
        self._check_infrequent_enabled()
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(X, force_all_finite=force_all_finite)
        self.n_features_in_ = n_features
        if self.categories != 'auto':
            if len(self.categories) != n_features:
                raise ValueError('Shape mismatch: if categories is an array, it has to be of shape (n_features,).')
        self.categories_ = []
        category_counts = []
        compute_counts = return_counts or self._infrequent_enabled
        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == 'auto':
                result = _unique(Xi, return_counts=compute_counts)
                if compute_counts:
                    cats, counts = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                if np.issubdtype(Xi.dtype, np.str_):
                    Xi_dtype = object
                else:
                    Xi_dtype = Xi.dtype
                cats = np.array(self.categories[i], dtype=Xi_dtype)
                if cats.dtype == object and isinstance(cats[0], bytes) and (Xi.dtype.kind != 'S'):
                    msg = f"In column {i}, the predefined categories have type 'bytes' which is incompatible with values of type '{type(Xi[0]).__name__}'."
                    raise ValueError(msg)
                for category in cats[:-1]:
                    if is_scalar_nan(category):
                        raise ValueError(f'Nan should be the last element in user provided categories, see categories {cats} in column #{i}')
                if cats.size != len(_unique(cats)):
                    msg = f'In column {i}, the predefined categories contain duplicate elements.'
                    raise ValueError(msg)
                if Xi.dtype.kind not in 'OUS':
                    sorted_cats = np.sort(cats)
                    error_msg = 'Unsorted categories are not supported for numerical categories'
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]):
                        raise ValueError(error_msg)
                if handle_unknown == 'error':
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = 'Found unknown categories {0} in column {1} during fit'.format(diff, i)
                        raise ValueError(msg)
                if compute_counts:
                    category_counts.append(_get_counts(Xi, cats))
            self.categories_.append(cats)
        output = {'n_samples': n_samples}
        if return_counts:
            output['category_counts'] = category_counts
        missing_indices = {}
        if return_and_ignore_missing_for_infrequent:
            for feature_idx, categories_for_idx in enumerate(self.categories_):
                if is_scalar_nan(categories_for_idx[-1]):
                    missing_indices[feature_idx] = categories_for_idx.size - 1
            output['missing_indices'] = missing_indices
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(n_samples, category_counts, missing_indices)
        return output

    def _transform(self, X, handle_unknown='error', force_all_finite=True, warn_on_unknown=False, ignore_category_indices=None):
        X_list, n_samples, n_features = self._check_X(X, force_all_finite=force_all_finite)
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)
        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)
            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    msg = 'Found unknown categories {0} in column {1} during transform'.format(diff, i)
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    X_mask[:, i] = valid_mask
                    if self.categories_[i].dtype.kind in ('U', 'S') and self.categories_[i].itemsize > Xi.itemsize:
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == 'O' and Xi.dtype.kind == 'U':
                        Xi = Xi.astype('O')
                    else:
                        Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(f'Found unknown categories in columns {columns_with_unknown} during transform. These unknown categories will be encoded as all zeros', UserWarning)
        self._map_infrequent_categories(X_int, X_mask, ignore_category_indices)
        return (X_int, X_mask)

    @property
    def infrequent_categories_(self):
        """Infrequent categories for each feature."""
        infrequent_indices = self._infrequent_indices
        return [None if indices is None else category[indices] for category, indices in zip(self.categories_, infrequent_indices)]

    def _check_infrequent_enabled(self):
        """
        This functions checks whether _infrequent_enabled is True or False.
        This has to be called after parameter validation in the fit function.
        """
        max_categories = getattr(self, 'max_categories', None)
        min_frequency = getattr(self, 'min_frequency', None)
        self._infrequent_enabled = max_categories is not None and max_categories >= 1 or min_frequency is not None

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        """Compute the infrequent indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Category counts.

        n_samples : int
            Number of samples.

        col_idx : int
            Index of the current category. Only used for the error message.

        Returns
        -------
        output : ndarray of shape (n_infrequent_categories,) or None
            If there are infrequent categories, indices of infrequent
            categories. Otherwise None.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)
        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            frequent_category_count = self.max_categories - 1
            if frequent_category_count == 0:
                infrequent_mask[:] = True
            else:
                smallest_levels = np.argsort(category_count, kind='mergesort')[:-frequent_category_count]
                infrequent_mask[smallest_levels] = True
        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(self, n_samples, category_counts, missing_indices):
        """Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is an array of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        .. versionadded:: 1.1

        Parameters
        ----------
        n_samples : int
            Number of samples in training set.
        category_counts: list of ndarray
            `category_counts[i]` is the category counts corresponding to
            `self.categories_[i]`.
        missing_indices : dict
            Dict mapping from feature_idx to category index with a missing value.
        """
        if missing_indices:
            category_counts_ = []
            for feature_idx, count in enumerate(category_counts):
                if feature_idx in missing_indices:
                    category_counts_.append(np.delete(count, missing_indices[feature_idx]))
                else:
                    category_counts_.append(count)
        else:
            category_counts_ = category_counts
        self._infrequent_indices = [self._identify_infrequent(category_count, n_samples, col_idx) for col_idx, category_count in enumerate(category_counts_)]
        self._default_to_infrequent_mappings = []
        for feature_idx, infreq_idx in enumerate(self._infrequent_indices):
            cats = self.categories_[feature_idx]
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue
            n_cats = len(cats)
            if feature_idx in missing_indices:
                n_cats -= 1
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats
            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)
            self._default_to_infrequent_mappings.append(mapping)

    def _map_infrequent_categories(self, X_int, X_mask, ignore_category_indices):
        """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.

        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.

        ignore_category_indices : dict
            Dictionary mapping from feature_idx to category index to ignore.
            Ignored indexes will not be grouped and the original ordinal encoding
            will remain.
        """
        if not self._infrequent_enabled:
            return
        ignore_category_indices = ignore_category_indices or {}
        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue
            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == 'infrequent_if_exist':
                X_mask[:, col_idx] = True
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue
            if i in ignore_category_indices:
                rows_to_update = X_int[:, i] != ignore_category_indices[i]
            else:
                rows_to_update = slice(None)
            X_int[rows_to_update, i] = np.take(mapping, X_int[rows_to_update, i])

    def _more_tags(self):
        return {'X_types': ['2darray', 'categorical'], 'allow_nan': True}