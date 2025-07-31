import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign.alignment.utils import (
    _check_input_arrays,
    _check_labels,
    _check_method,
    _check_target,
    _fit_template,
    _map_to_target,
)


class GroupAlignment(BaseEstimator, TransformerMixin):
    """Performs group-level alignment of various subject data.

    This class aligns multiple subjects' data either to a computed template
    or to a specific target subject's data. It supports various alignment
    methods and can process data in parallel.

    Parameters
    ----------
    method : str or a `BaseAlignment` instance, default="identity"
        The alignment method to use. Supported methods depend on the
        underlying alignment implementation.
    labels : array-like or None, default=None
        Describes each voxel label's in the case of non-overlapping parcels.
        If provided, local alignments can be performed in parallel.
        If None, global alignment is performed across all features.
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.
    verbose : int, default=0
        Verbosity level. Higher values provide more detailed output.
    n_iter : int, default=2
        Number of iterations for the template alignment algorithm.
    scale_template : bool, default=False
        Whether to scale the template during template alignment.

    Attributes
    ----------
    labels_ : array-like
        Validated labels used during fitting.
    method_ : str
        Validated alignment method used during fitting.
    fit_ : list
        List of fitted alignment estimators, one per subject.
    template : array-like or None
        Computed template for template alignment. None for pairwise alignment.

    Examples
    --------
    >>> # Template alignment
    >>> aligner = GroupAlignment(method="procrustes", n_iter=3)
    >>> aligner.fit(alignment_dict)
    >>> aligned_data = aligner.transform(testing_dict)

    >>> # Pairwise alignment to target
    >>> target_data = np.array([[1, 2], [3, 4]])
    >>> aligner = GroupAlignment(method="procrustes")
    >>> aligner.fit(alignment_dict, y=target_data)
    >>> aligned_data = aligner.transform(testing_dict)
    """

    def __init__(
        self,
        method="identity",
        labels=None,
        n_jobs=1,
        verbose=0,
        n_iter=2,
        scale_template=False,
    ):
        self.method = method
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.template = None

    def fit(self, X, y="template") -> None:
        """Fit the group alignment model to the data.

        Parameters
        ----------
        X : dict of array-like
            Dictionary where keys are subject identifiers and values are
            arrays of subject data. Each array should have the same number of
            samples and features.
        y : str or array-like, default="template"
            Target for alignment. If "template", performs template alignment where
            a template is computed from all subjects. If array-like, performs
            pairwise alignment to the specified target data.
        """
        # Validate input data
        self.subject_keys_, X_ = _check_input_arrays(X)
        y_ = _check_target(X_[0], y)
        self.labels_ = _check_labels(X_[0], self.labels)
        self.method_ = _check_method(self.method)

        if y_ is None:  # Template alignment
            fit_, self.template = _fit_template(
                X_,
                self.method_,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
                self.n_iter,
                self.scale_template,
            )
        else:  # Pairwise alignment
            fit_ = _map_to_target(
                X_,
                y_,
                self.method_,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
            )

        self.fitted_estimators = dict(zip(self.subject_keys_, fit_))
        return self

    def _transform_one_array(self, X, estimator):
        """Transform a single subject's data using a fitted estimator.

        Parameters
        ----------
        X : array-like
            Subject data to transform. Should have the same number of
            voxels/columns as the data used during fitting.
        estimator : A fitted alignment estimator
            The fitted alignment estimator to use for transformation.

        Returns
        -------
        array-like
            Aligned subject data.

        Raises
        ------
        ValueError
            If the estimator has not been fitted yet.
        """
        # Check if estimator is fitted
        if not hasattr(estimator, "fit"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        return estimator.transform(X)

    def transform(self, X):
        """Transform the input arrays using the fitted model.

        Parameters
        ----------
        X : dict of array-like
            Dictionary where keys are subject identifiers and values are
            array of subject data. Each array should have the same number of
            samples and features.

        Returns
        -------
        dict of array-like
            Dictionary with transformed subject data.
        """
        if not hasattr(self, "fitted_estimators"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        # Check if the keys are valid
        keys = list(X.keys())
        if not all(key in self.subject_keys_ for key in keys):
            raise ValueError(
                "Some subjects identifier are not present in the fitted model. "
                "Please check the input keys."
            )

        return {
            key: self._transform_one_array(X[key], self.fitted_estimators[key])
            for key in keys
        }

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no 'fit_transform' attribute"
        )
