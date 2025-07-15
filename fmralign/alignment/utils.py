import numpy as np
from sklearn.base import clone
from fmralign.methods.piecewise import PiecewiseAlignment


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


def _check_method(method, labels):
    return method


def _map_to_target(
    X,
    target_data,
    method,
    labels,
    n_jobs,
    verbose,
):
    """Fit each subject's data to a target using the specified method.

    Parameters
    ----------
    X : list of ndarray
        List of subject data arrays, where each array is of shape (n_samples, n_features).
    target_data : ndarray
        The target data array to which each subject's data will be fitted.
    method : an instance of any class derived from `BaseAlignment`
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
    for subject_data in X:
        if n_labels is not None and n_labels > 1:
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
    n_jobs,
    verbose,
    n_iter=2,
    scale_template=False,
):
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
