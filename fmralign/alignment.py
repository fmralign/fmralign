from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from fmralign._utils import _check_labels
import numpy as np
from fmralign.template.utils import _rescaled_euclidean_mean


def fit_to_target(X, target_data, method, labels, n_jobs, verbose):
    """Fit each subject's data to a target using the specified method.

    Parameters
    ----------
    X : list of ndarray
        List of subject data arrays, where each array is of shape (n_samples, n_features).
    target_data : ndarray
        The target data array to which each subject's data will be fitted.
    method : str or an instance of any class derived from `BaseAlignment`
        Algorithm used to perform alignment between sources and target.
    labels : np.ndarray or list of int
        Labels for the parcellation of the data. If only one label is present,
        the whole brain method is used.
        If multiple labels are present, the method will patch the parcels estimators
        in a big whole brain estimator.
    n_jobs : int
        Number of jobs to run in parallel. If -1, all CPUs are used.
        If 1, no parallel computing code is used at all.
    verbose : int
        Verbosity level.

    Returns
    -------
    list of fitted estimators
    """
    n_labels = len(np.unique(labels))
    fitted_estimators = []
    for sub in X:
        if n_labels == 1:
            # If only one label, use the whole brain method
            fitted_estimator = ...
            fitted_estimators.append(fitted_estimator)
        else:
            # Patch the parcels estimators in a big whole brain estimator
            fitted_estimator = ...
            fitted_estimators.append(fitted_estimator)
    return fitted_estimators


def fit_template(
    X, method, labels, n_jobs, verbose, n_iter=2, scale_template=False
):
    # Template is initialized as the mean of all subjects
    aligned_data = X
    # Fit template alignment
    for _ in range(n_iter):
        template = _rescaled_euclidean_mean(aligned_data, scale_template)
        fit_ = fit_to_target(
            X,
            template,
            method,
            labels,
            n_jobs,
            verbose,
        )
        aligned_data = [fit_[i].transform(X[i]) for i in range(len(X))]
    return fit_, template


def fit_pairwise(X, target, method, labels, n_jobs, verbose):
    return fit_to_target(X, target, method, labels, n_jobs, verbose)


class Alignment(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method="identity",
        target=None,
        labels=None,
        n_jobs=1,
        verbose=0,
        n_iter=2,
        scale_template=False,
    ):
        self.method = method
        self.target = target
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_iter = n_iter
        self.template = None

    def fit(self, X, y=None) -> None:
        self.labels_ = _check_labels(X[0], self.labels, verbose=self.verbose)

        if self.target is None:  # Template alignment
            self.fit_, self.template = fit_template(
                X,
                self.method,
                self.labels_,
                self.n_iter,
                self.n_jobs,
                self.verbose,
            )
        elif isinstance(self.target, np.ndarray):  # Pairwise alignment
            self.fit_ = fit_pairwise(
                X,
                self.target,
                self.method,
                self.labels_,
                self.n_jobs,
                self.verbose,
            )
        else:
            raise ValueError(
                "Target must be an integer index of the subject "
                "or None for template alignment."
            )

    def _transform_one_array(self, X, method):
        # Check if method is fitted
        if not hasattr(method, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        return method.transform(X)

    def transform(self, X, subject_indices):
        return [
            self._transform_one_array(X[i], self.fit_[i])
            for i in subject_indices
        ]

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no 'fit_transform' method"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.parcel_masker.get_labels()
            parcellation_img = self.parcel_masker.get_parcellation_img()
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
