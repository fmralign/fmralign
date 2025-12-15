from sklearn.base import BaseEstimator, TransformerMixin

from fmralign.alignment.utils import (
    _check_labels,
    _check_method,
    _check_target,
    _map_to_target,
)


class PairwiseAlignment(BaseEstimator, TransformerMixin):
    """Performs pairwise alignment between two subjects.

    This class performs source-to-target alignment of two subjects' data.
    It supports various alignment methods and can process data in parallel.

    Parameters
    ----------
    method : str or a `BaseAlignment` instance, default="identity"
        The alignment method to use. It can be a string representing the method name
        or an instance of a class derived from `BaseAlignment`. Available methods include:
        ["identity", "procrustes", "ot", "ridge"].
    labels : array-like or None, default=None
        Describes each voxel label's in the case of non-overlapping parcels.
        If provided, local alignments can be performed in parallel.
        If None, global alignment is performed across all features.
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.
    verbose : int, default=0
        Verbosity level. Higher values provide more detailed output.

    Attributes
    ----------
    labels_ : array-like
        Validated labels used during fitting.
    method_ : str
        Validated alignment method used during fitting.
    fitted_estimator : `BaseAlignment` object
        List of fitted alignment estimators, one per subject.

    Examples
    --------
    >>> from fmralign import PairwiseAlignment
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(10, 5)
    >>> test_data = np.random.rand(8, 5)
    >>> aligner = PairwiseAlignment(method="procrustes")
    >>> aligner.fit(X, Y)
    >>> aligned_data = aligner.transform(test_data)
    """

    def __init__(
        self,
        method="identity",
        labels=None,
        n_jobs=1,
        verbose=0,
    ):
        self.method = method
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, Y):
        """Fit the pairwise alignment model to the data.

        Parameters
        ----------
        X : array-like
            Source data for alignment of shape (n_samples, n_features).
        Y : array-like
            Target data for alignment of shape (n_samples, n_features).
        """
        # Validate input data
        X_ = [X]
        Y_ = _check_target(X, Y)
        self.labels_ = _check_labels(X, self.labels)
        self.method_ = _check_method(self.method)

        fit_ = _map_to_target(
            X_,
            Y_,
            self.method_,
            self.labels_,
            self.n_jobs,
            self.verbose,
        )

        self.fitted_estimator = fit_[0]
        return self

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
        if not hasattr(self, "fitted_estimator"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        return self.fitted_estimator.transform(X)

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no 'fit_transform' attribute"
        )
