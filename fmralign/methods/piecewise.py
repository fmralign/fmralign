import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from fmralign.methods.base import BaseAlignment


def _fit_one_piece(X, Y, method):
    """
    Fit a single piece of data to the target.

    Parameters
    ----------
    X : ndarray
        Source data of shape (n_samples, n_features).
    Y : ndarray
        Target data of shape (n_samples, n_features).

    Returns
    -------
    ndarray
        Fitted piece of data.
    """
    # Clone the method to avoid modifying the original instance
    estimator = clone(method)
    estimator.fit(X, Y)
    return estimator


def _transform_one_piece(X, estimator):
    """
    Transform a single piece of data using the fitted estimator.

    Parameters
    ----------
    X : ndarray
        Source data of shape (n_samples, n_features).

    Returns
    -------
    ndarray
        Transformed piece of data.
    """
    return estimator.transform(X)


def _array_to_list(arr, labels):
    """
    Convert a 2D array to a list of arrays based on labels.

    Parameters
    ----------
    arr : ndarray
        2D array of shape (n_samples, n_features).
    labels : list or ndarray
        Labels for each sample.

    Returns
    -------
    list of ndarray
        List of arrays corresponding to each label.
    """
    unique_labels = np.unique(labels)
    return [arr[:, labels == label] for label in unique_labels]


def _list_to_array(lst, labels):
    """
    Convert a list of arrays back to a 2D array based on labels.
    Parameters
    ----------
    lst : list of ndarray
        List of arrays, where each array corresponds to a unique label.
    labels : list or ndarray
        Labels for each sample.
    Returns
    -------
    ndarray
        2D array of shape (n_samples, n_features) where each column corresponds to
        a unique label from the input list.
    """
    unique_labels = np.unique(labels)
    n_features = len(labels)
    n_samples = lst[0].shape[0]
    data = np.zeros((n_samples, n_features))
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        data[:, labels == label] = lst[i]
    return data


class PiecewiseAlignment(BaseAlignment):
    """Class for piecewise alignment.

    This class allows for fitting an alignment method in a piecewise manner,
    where each piece corresponds to a unique label in the provided labels array.

    It uses parallel processing to fit the alignment method to each piece of data
    and agregates the results in a estimator at the whole-brain level.

    Parameters
    ----------
    method : `BaseAlignment`
        Alignment method to be used for each piece.
    labels : 1D np.ndarray
        Labels used to determine how to split the data between parcels.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    def __init__(self, method, labels, n_jobs=1, verbose=0):
        super().__init__()
        self.n_jobs = n_jobs
        self.method = method
        self.labels = labels
        self.verbose = verbose

    def fit(self, X, Y):
        """Fit the alignment method to the source and target data.

        Parameters
        ----------
        X : np.ndarray
            Source data of shape (n_samples, n_features).
        Y : np.ndarray
            Target data of shape (n_samples, n_features).
        """
        X_ = _array_to_list(X, self.labels)
        Y_ = _array_to_list(Y, self.labels)
        self.fit_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_one_piece)(X_[i], Y_[i], self.method)
            for i in range(len(X_))
        )
        return self

    def transform(self, X):
        """
        Transform X using the fitted estimator.

        Parameters
        ----------
        X : ndarray
            Source data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed data of shape (n_samples, n_features).
        """
        X_ = _array_to_list(X, self.labels)
        piecewise_transforms = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose
        )(
            delayed(_transform_one_piece)(X_[i], self.fit_[i])
            for i in range(len(X_))
        )
        return _list_to_array(piecewise_transforms, self.labels)
