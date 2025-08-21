from scipy import linalg

from fmralign.methods.base import BaseAlignment


class DetSRM(BaseAlignment):
    """
    Compute a orthogonal mixing matrix R and a scaling sc.
    These are calculated such that Frobenius norm ||sc RX - Y||^2 is minimized.

    Parameters
    -----------
    scaling : boolean, optional
        Determines whether a scaling parameter is applied to improve transform.

    Attributes
    -----------
    R : ndarray (n_features, n_features)
        Optimal orthogonal transform
    scale: float,
               inferred scaling parameter
    """

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, S):
        """
        Fit orthogonal R s.t. ||sc XR - Y||^2

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            Source data
        S: (n_samples, n_components) nd array
            Shared response
        """
        U, _, V = linalg.svd(S.T @ X, full_matrices=False)
        self.W = U @ V
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X @ self.W.T
