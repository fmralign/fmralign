import numpy as np

from fmralign.methods.base import BaseAlignment


def scaled_procrustes(X, Y, scaling=False, primal=None):
    r"""
    Batched version. Compute a mixing matrix R and a scaling sc such that Frobenius norm
    :math:`||sc RX - Y||^2` is minimized and R is an orthogonal matrix, for each
    batch element independently.

    Parameters
    ----------
    X: (b, n_samples, n_features) nd array
        source data
    Y: (b, n_samples, n_features) nd array
        target data
    scaling: bool
        If scaling is true, computes a floating scaling parameter sc such that:
        ||sc * RX - Y||^2 is minimized and
        - R is an orthogonal matrix
        - sc is a scalar
        If scaling is false sc is set to 1
    primal: bool or None, optional,
         Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
         if None primal is used iff n_features <= n_timeframes
    Returns
    -------
    R: (b, n_features, n_features) nd array
        transformation matrix
    sc: (b,) nd array
        scaling parameter
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)

    b, n_samples, n_features = X.shape
    X_norm = np.linalg.norm(X.reshape(b, -1), axis=1)

    if primal is None:
        primal = n_samples >= n_features
    if primal:
        A = Y.transpose(0, 2, 1) @ X
        if A.shape[1] == A.shape[2]:
            A += 1.0e-18 * np.eye(A.shape[1])
        U, s, V = np.linalg.svd(A, full_matrices=0)
        R = U @ V
    else:  # "dual" mode
        Uy, sy, Vy = np.linalg.svd(Y, full_matrices=0)
        Ux, sx, Vx = np.linalg.svd(X, full_matrices=0)
        A = np.einsum(
            "bij,bjk->bik",
            sy[..., None] * np.eye(sy.shape[-1]),
            Uy.transpose(0, 2, 1),
        )
        A = A @ Ux @ (sx[..., None] * np.eye(sx.shape[-1]))
        U, s, V = np.linalg.svd(A)
        R = Vy.transpose(0, 2, 1) @ U @ V @ Vx
    if scaling:
        sc = s.sum(axis=-1) / (X_norm**2)
    else:
        sc = np.ones(b)

    return R.transpose(0, 2, 1), sc


class Procrustes(BaseAlignment):
    r"""
    Compute a orthogonal mixing matrix R and a scaling sc.
    These are calculated such that Frobenius norm :math:`||sc RX - Y||^2` is minimized.

    Parameters
    ----------
    scaling : boolean, optional
        Determines whether a scaling parameter is applied to improve transform.

    Attributes
    ----------
    R : ndarray (n_features, n_features)
        Optimal orthogonal transform
    scale: float,
               inferred scaling parameter
    """

    def __init__(self, scaling=True):
        self.scaling = scaling
        self.scale = 1

    def fit(self, X, Y):
        r"""
        Fit orthogonal R s.t. :math:`||sc XR - Y||^2`

        Parameters
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        R, sc = scaled_procrustes(X, Y, scaling=self.scaling)
        self.scale = sc
        self.R = R
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X @ self.R * self.scale[:, None, None]
