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
        verbose=False,
        device="cpu",
        **kwargs,
    ):
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
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
        R_torch = torch.tensor(self.R, device=self.device, dtype=torch.float32)
        X_torch = torch.from_numpy(X).to(torch.float32).to(self.device)
        return (X_torch @ R_torch * n_voxels).cpu().numpy()


class SpectralOT(OptimalTransport):
    """
    Compute the optimal coupling between X and Y using an anatomy-aware
    cost matrix that combines functional and harmonic distances.

    Parameters
    ----------
    evecs : (k, n_features) nd array
        Harmonic embedding of the geometry, first k eigenmodes of
        the Laplace-Beltrami operator.
    alpha : float
        Trade-off parameter controlling the balance between functional
        data and the harmonic embedding `evecs`. Values should lie in the
        interval [0, 1], where higher values put more weight on the anatomy.
        Defaults to 0.5.
    reg : float (optional)
        Strength of the entropic regularization. Defaults to 0.01.
    max_iter : int (optional)
        Maximum number of iterations. Defaults to 1000.
    tol : float (optional)
        Tolerance for stopping criterion. Defaults to 1e-7.
    verbose : bool (optional)
        Allow verbose output. Defaults to False.
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
        evecs,
        alpha=0.5,
        reg=1e-2,
        max_iter=1000,
        tol=1e-7,
        verbose=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            reg=reg,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            device=device,
        )

        self.alpha = alpha
        self.evecs = evecs
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
            np.ascontiguousarray(X.T),
            device=self.device,
            dtype=torch.float32,
        )
        Y_torch = torch.tensor(
            np.ascontiguousarray(Y.T),
            device=self.device,
            dtype=torch.float32,
        )
        evecs_torch = torch.tensor(
            np.ascontiguousarray(self.evecs.T),
            device=self.device,
            dtype=torch.float32,
        )

        M_func = ot.dist(X_torch, Y_torch)
        M_geom = ot.dist(evecs_torch)

        # Normalize both cost matrices to have the same scale
        M_func_normalized = ot.utils.cost_normalization(M_func, "max")
        M_geom_normalized = ot.utils.cost_normalization(M_geom, "max")
        M_normalized = (
            1 - self.alpha
        ) * M_func_normalized + self.alpha * M_geom_normalized

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
