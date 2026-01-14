import numpy as np
import ot
import torch
from pykeops.torch import LazyTensor

from fmralign.methods.base import BaseAlignment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OptimalTransport(BaseAlignment):
    """
    Compute the optimal coupling between X and Y with entropic regularization,
    using the pure Python POT (https://pythonot.github.io/) package.

    Parameters
    ----------
    reg : float (optional)
        Strength of the entropic regularization. Defaults to 0.01.
    max_iter : int (optional)
        Maximum number of iterations. Defaults to 1000.
    tol : float (optional)
        Tolerance for stopping criterion. Defaults to 1e-7.
    verbose : bool (optional)
        Allow verbose output. Defaults to False.
    scaling : float
        Scaling parameter for GeomLoss solver when n_features > 1000.
        Defaults to 0.95.
    alpha : float
        Trade-off parameter controlling the balance between functional data and the
        geometric embedding `evecs`. Values should lie in the interval [0, 1], where
        smaller values put more weight on the geometry.
        Defaults to 0.1.
    evecs : (k, n_features) nd array or None
        Geometric embedding of the data to be used as additional features
        during alignment. If None, only functional data is used.
        Defaults to None.
    backend : str
        Backend to use for OT computation. Either "pot" or "geomloss".
        Defaults to "pot".
    kwargs : dict
        Additional arguments to be passed to the OT solver.

    Attributes
    ----------
    R : (n_features, n_features) nd array or LazyTensor
        Transport plan computed during fitting.
    """

    def __init__(
        self,
        reg=1e-2,
        max_iter=1000,
        tol=1e-7,
        scaling=0.95,
        alpha=0.1,
        evecs=None,
        backend="pot",
        verbose=False,
        **kwargs,
    ):
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.scaling = scaling
        self.alpha = alpha
        self.evecs = evecs
        self.backend = backend
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, Y):
        """

        Parameters
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """

        if self.evecs is not None:
            X = np.vstack(
                [
                    self.alpha * X,
                    (1 - self.alpha) * self.evecs,
                ]
            )
            Y = np.vstack(
                [
                    self.alpha * Y,
                    (1 - self.alpha) * self.evecs,
                ]
            )

        if self.backend == "pot":
            res = ot.solve_sample(
                X.T,
                Y.T,
                reg=self.reg,
                tol=self.tol,
                max_iter=self.max_iter,
                lazy=False,
                verbose=self.verbose,
                **self.kwargs,
            )

            self.R = res.plan

        elif self.backend == "geomloss":
            X_torch = torch.tensor(np.ascontiguousarray(X.T), device=DEVICE)
            Y_torch = torch.tensor(np.ascontiguousarray(Y.T), device=DEVICE)
            res = ot.solve_sample(
                X_torch,
                Y_torch,
                reg=self.reg,
                lazy=True,
                method="geomloss",
                verbose=self.verbose,
                scaling=self.scaling,
                **self.kwargs,
            )

            f, g = res.potentials
            blur = float(res.lazy_plan.blur)
            X_i = LazyTensor(X_torch[:, None, :])
            Y_j = LazyTensor(Y_torch[None, :, :])
            F = LazyTensor(f[:, None, None])
            G = LazyTensor(g[None, :, None])
            C = ((X_i - Y_j) ** 2).sum(-1) / 2
            self.R = (F + G - C / (blur**2)).exp() / (
                X_torch.shape[0] * Y_torch.shape[0]
            )

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        n_voxels = X.shape[1]
        if self.backend == "geomloss":
            X_torch = torch.tensor(np.ascontiguousarray(X.T), device=DEVICE)
            X_i = LazyTensor(X_torch[:, None, :])
            X_aligned = (X_i * self.R).sum(axis=0) * n_voxels
            return X_aligned.cpu().numpy().T
        else:
            return X @ self.R * n_voxels
