import numpy as np
from nilearn import signal


def get_connectivity_features(data, labels):
    """Compute connectivity features for a single subject.

    Parameters
    ----------
    data: array-like of shape (n_samples, n_voxels)
    labels: array-like of shape (n_voxels,)
        Labels for connectivity seeds.

    Returns
    -------
    correlation_features: array-like of shape (n_labels, n_voxels)
        Connectivity features computed as the correlation between
        averaged signals within each parcel and the original data.
    """

    # Standardize the data
    standardized_data = signal.standardize_signal(
        data, detrend=True, standardize=True
    )

    # Average the signals within each parcel
    averaged_signals = np.stack(
        [
            standardized_data[:, labels == lbl].mean(axis=1)
            for lbl in np.unique(labels)
        ],
        axis=1,
    )

    # Compute the correlation features (n_labels x n_voxels)
    correlation_features = (
        averaged_signals.T @ standardized_data / averaged_signals.shape[0]
    )
    correlation_features = np.nan_to_num(correlation_features)
    return correlation_features
