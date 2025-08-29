from scipy import linalg

from fmralign.methods.base import BaseAlignment


class DetSRM(BaseAlignment):
    """Compute the alignment from one subjects to the shared latent response.

    Parameters
    ----------
    n_components: int
        Number of shared components. Defaults to 20.

    Attributes
    ----------
    W : (n_components, n_voxels) ndarray
        Optimal mixing matrix
    """

    def __init__(self, n_components=20):
        self.n_components = n_components

    def fit(self, X, S):
        """
        Fit orthogonal W s.t. ||X - SW||^2 is minimized

        Parameters
        -----------
        X: (n_samples, n_features) ndarray
            Source data
        S: (n_samples, n_components) ndarray
            Shared response
        """
        U, _, V = linalg.svd((S.T @ X).T, full_matrices=False)
        self.Wt = U @ V
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X @ self.Wt
