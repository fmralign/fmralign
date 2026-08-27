import numpy as np

from fmralign.methods.base import BaseAlignment


class DetSRM(BaseAlignment):
    """Batched version. Compute the alignment from subjects to the shared latent response.
    Parameters
    ----------
    n_components: int
        Number of shared components. Defaults to 20.
    Attributes
    ----------
    Wt : (b, n_components, n_voxels) ndarray
        Optimal mixing matrix, per batch element
    """

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, S):
        r"""
        Fit orthogonal W s.t. :math:`||X - SW||^2` is minimized, independently
        for each batch element.
        Parameters
        ----------
        X: (b, n_samples, n_features) ndarray
            Source data
        S: (b, n_samples, n_components) ndarray
            Shared response
        """
        U, _, V = np.linalg.svd(
            (S.transpose(0, 2, 1) @ X).transpose(0, 2, 1), full_matrices=False
        )
        self.Wt = U @ V
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X @ self.Wt
