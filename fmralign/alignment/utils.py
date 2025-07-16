import numpy as np
from sklearn.base import clone
from fmralign.methods.piecewise import PiecewiseAlignment
import warnings
from fmralign.methods import (
    Identity,
    OptimalTransport,
    SparseUOT,
    ScaledOrthogonal,
    RidgeAlignment,
)


def _rescaled_euclidean_mean(subjects_data, scale_average=False):
    """
    Make the Euclidian average of `numpy.ndarray`.

    Parameters
    ----------
    subjects_data: `list` of `numpy.ndarray`
        Each element of the list is the data for one subject.
    scale_average: boolean
        If true, average is rescaled so that it keeps the same norm as the
        average of training images.

    Returns
    -------
    average_data: ndarray
        Average of imgs, with same shape as one img
    """
    average_data = np.mean(subjects_data, axis=0)
    scale = 1
    if scale_average:
        X_norm = 0
        for data in subjects_data:
            X_norm += np.linalg.norm(data)
        X_norm /= len(subjects_data)
        scale = X_norm / np.linalg.norm(average_data)
    average_data *= scale

    return average_data


def _check_input_arrays(X):
    """Check if the input data is a list of 2D numpy arrays.

    Parameters
    ----------
    X : list of ndarray
        List of subject data arrays, where each array is of shape (n_samples, n_features).

    Returns
    -------
    X : list of ndarray
        The validated list of subject data arrays.
    """
    if not isinstance(X, list):
        raise ValueError("Input data must be a list of arrays.")
    if len(X) == 0:
        raise ValueError("Input data list cannot be empty.")
    if not all(isinstance(x, np.ndarray) for x in X):
        raise ValueError(
            "All elements in the input list must be numpy arrays."
        )
    if not all(x.ndim == 2 for x in X):
        raise ValueError("All arrays in the input list must be 2D arrays.")
    if not all(x.shape[0] == X[0].shape[0] for x in X):
        raise ValueError(
            "All arrays in the input list must have the same number of samples."
        )
    if not all(x.shape[1] == X[0].shape[1] for x in X):
        raise ValueError(
            "All arrays in the input list must have the same number of features."
        )
    return X


def _check_method(method):
    """Check if the method is part of the valid methods and return the corresponding instance.

    Parameters
    ----------
    method : str or `BaseAlignment`
        The alignment method to use. It can be a string representing the method name
        or an instance of a class derived from `BaseAlignment`.

    Returns
    -------
    BaseAlignment
        An instance of the specified alignment method.

    Raises
    ------
    ValueError
        If the method is not recognized or not an instance of a valid alignment method.
    """
    # Check if the method is part of the valid methods
    valid_methods = {
        "identity": Identity(),
        "ot": OptimalTransport(),
        "sparse_uot": SparseUOT(),
        "scaled_orthogonal": ScaledOrthogonal(),
        "ridge": RidgeAlignment(),
    }
    # If method is a string, convert it to the corresponding class instance
    if isinstance(method, str):
        method = valid_methods.get(method.lower())
        if method is None:
            raise ValueError(
                f"Method '{method}' is not recognized. "
                f"Valid methods are: {valid_methods.keys()}"
            )

    return method


def _check_labels(X, labels=None, threshold=1000, verbose=0):
    """Check if any parcels are bigger than set threshold.

    Parameters
    ----------
    X : ndarray
        The data array of shape (n_samples, n_features).
    labels : 1D np.ndarray or None
        Labels for the parcellation of the data.
    threshold : int
        The threshold for the maximum size of a parcel. If any parcel exceeds this size,
        a warning will be raised.
    verbose : int
        Verbosity level. If greater than 0, prints the sizes of the parcels.

    Returns
    -------
    labels : 1D np.ndarray
        The labels array, converted to integer type if necessary.
    """
    if labels is None:
        # If no labels are provided, create a single label for the whole brain
        labels = np.ones(X.shape[1], dtype=int)
    else:
        if len(labels) != X.shape[1]:
            raise ValueError(
                "The length of labels must match the number of features in the data."
            )
        if labels.ndim != 1:
            raise ValueError("Labels must be a 1D array.")

        unique_labels, counts = np.unique(labels, return_counts=True)

        if verbose > 0:
            print(
                f"The alignment will be applied on parcels of sizes {counts}"
            )

        if not all(count < threshold for count in counts):
            warning = (
                "\n Some parcels are more than 1000 voxels wide it can slow down alignment,"
                "especially optimal_transport :"
            )
            for i in range(len(counts)):
                if counts[i] > threshold:
                    warning += (
                        f"\n parcel {unique_labels[i]} : {counts[i]} voxels"
                    )
            warnings.warn(warning)

        # If labels are not integer type, convert them to int
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(int)
            warnings.warn("Labels were not integer type, converted to int.")

    return labels


def _map_to_target(
    X,
    target_data,
    method,
    labels,
    n_jobs=1,
    verbose=0,
):
    """Fit each subject's data to a target using the specified method.

    Parameters
    ----------
    X : list of 2D ndarray
        List of subject data arrays, where each array is of shape (n_samples, n_features).
    target_data : ndarray
        The target data array to which each subject's data will be fitted.
    method : an instance of any class derived from `BaseAlignment`
        Algorithm used to perform alignment between sources and target.
    labels : 1D np.ndarray
        Labels for the parcellation of the data. If only one label is present,
        the whole brain method is used.
        If multiple labels are present, the method will patch the parcels estimators
        in a big whole brain estimator.
    n_jobs : int, optional
        Number of jobs to run in parallel. If -1, all CPUs are used.
        If 1, no parallel computing code is used at all, by default 1
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    list of fitted estimators
    """
    n_labels = len(np.unique(labels))
    fitted_estimators = []
    for subject_data in X:
        if n_labels > 1:
            estimator = PiecewiseAlignment(
                method=method,
                labels=labels,
                n_jobs=n_jobs,
                verbose=max(verbose - 1, 0),
            )
            estimator.fit(subject_data, target_data)
            fitted_estimators.append(estimator)
        else:
            estimator = clone(method)
            estimator.fit(subject_data, target_data)
            fitted_estimators.append(estimator)

    return fitted_estimators


def _fit_template(
    X,
    method,
    labels,
    n_jobs=1,
    verbose=0,
    n_iter=2,
    scale_template=False,
):
    """Fit a template to the target data using the specified method.

    Parameters
    ----------
    X : list of 2D ndarray
        List of subject data arrays, where each array is of shape (n_samples, n_features).
    method : an instance of any class derived from `BaseAlignment`
        Algorithm used to perform alignment between sources and target.
    labels : 1D np.ndarray
        Labels for the parcellation of the data. If only one label is present,
        the whole brain method is used.
        If multiple labels are present, the method will patch the parcels estimators
        in a big whole brain estimator.
    n_jobs : int, optional
        Number of jobs to run in parallel. If -1, all CPUs are used.
        If 1, no parallel computing code is used at all, by default 1
    verbose : int, optional
        Verbosity level, by default 0
    n_iter : int, optional
        Number of template updates, by default 2
    scale_template : bool, optional
        Rescale the template at each feature update, by default False

    Returns
    -------
    fit_ : list of fitted estimators
    template : ndarray
        The template data array.
    """
    # Template is initialized as the mean of all subjects
    aligned_data = X
    # Fit template alignment
    for _ in range(n_iter):
        template = _rescaled_euclidean_mean(aligned_data, scale_template)
        fit_ = _map_to_target(
            X,
            template,
            method,
            labels,
            n_jobs,
            max(verbose - 1, 0),
        )
        aligned_data = [fit_[i].transform(X[i]) for i in range(len(X))]
    return fit_, template
