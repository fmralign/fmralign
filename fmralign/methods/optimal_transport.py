import numpy as np
import ot
import torch

from fmralign.methods.base import BaseAlignment


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
    device : str
        Torch compatible device. Defaults to "cpu".
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
        verbose=False,
        device="cpu",
        **kwargs,
    ):
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.scaling = scaling
        self.verbose = verbose
        self.device = device
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
        X_torch = torch.tensor(
            np.ascontiguousarray(X.T), device=self.device, dtype=torch.float32
        )
        Y_torch = torch.tensor(
            np.ascontiguousarray(Y.T), device=self.device, dtype=torch.float32
        )

        M = ot.dist(X_torch, Y_torch)
        M_normalized = ot.utils.cost_normalization(M, "max")

        res = ot.solve(
            M=M_normalized,
            reg=self.reg,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            grad="detach",
            **self.kwargs,
        )

        # Store the transport plan on CPU as a numpy array
        self.R = res.plan.cpu().numpy()

        return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        n_voxels = self.R.shape[1]
        X_torch = torch.from_numpy(X).to(torch.float32).to(self.device)
        return (X_torch @ self.R.to(self.device) * n_voxels).cpu().numpy()
