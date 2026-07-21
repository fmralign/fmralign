import importlib.util

import numpy as np

from fmralign.methods.base import BaseAlignment

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_POT = importlib.util.find_spec("ot") is not None
_HAS_GEOMLOSS = importlib.util.find_spec("geomloss") is not None


def _require(checks, extra):
    """Raise a clear error listing any missing packages for a given extra.

    Parameters
    ----------
    checks : list of (bool, str) tuples
        Each tuple is (is_available, package_name).
    extra : str
        The pip extra that installs the missing package(s).
    """
    missing = [name for available, name in checks if not available]
    if missing:
        raise ImportError(
            f"{', '.join(missing)} required but not installed. "
            f"Install with `pip install fmralign[{extra}]`."
        )


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
    R : (n_features, n_features) nd array
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
        _require([(_HAS_TORCH, "torch"), (_HAS_POT, "POT")], extra="ot_pot")
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
        import ot
        import torch

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
            method="sinkhorn_log",
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
        import torch

        n_voxels = self.R.shape[1]
        R_torch = torch.tensor(self.R, device=self.device, dtype=torch.float32)
        X_torch = torch.from_numpy(X).to(torch.float32).to(self.device)
        return (X_torch @ R_torch * n_voxels).cpu().numpy()


class SpectralOT(BaseAlignment):
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
    backend : str
        Backend to use for the OT solver. Can be either "pot"
        (Python Optimal Transport) or "geomloss" (GeomLoss library).
        Defaults to "pot".
    device : str
        Torch compatible device. Defaults to "cpu".
    kwargs : dict
        Additional arguments to be passed to the OT solver.

    Attributes
    ----------
    R : (n_features, n_features) nd array (pot) or LinearOperator (geomloss)
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
        backend="pot",
        device="cpu",
        **kwargs,
    ):
        if backend not in ("pot", "geomloss"):
            raise ValueError(
                f"Unknown backend {backend!r}. Valid backends are 'pot', 'geomloss'."
            )

        if backend == "pot":
            _require(
                [(_HAS_TORCH, "torch"), (_HAS_POT, "POT")], extra="ot_pot"
            )
        else:
            _require(
                [(_HAS_TORCH, "torch"), (_HAS_GEOMLOSS, "geomloss")],
                extra="ot_geomloss",
            )

        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.alpha = alpha
        self.evecs = evecs
        self.kwargs = kwargs
        self.backend = backend

    def fit(self, X, Y):
        """

        Parameters
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        import torch

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

        if self.backend == "pot":
            self.R = _pot_solver(
                X_torch,
                Y_torch,
                evecs_torch,
                self.reg,
                self.alpha,
                self.max_iter,
            )

        elif self.backend == "geomloss":
            self.R = _geomloss_solver(
                X_torch,
                Y_torch,
                evecs_torch,
                self.reg,
                self.alpha,
                self.max_iter,
            )
        else:
            raise ValueError(
                f"Unknown backend {self.backend}. Valid backends are 'geomloss', 'pot'."
            )

        return self

    def transform(self, X):
        import torch

        if self.backend == "pot":
            n_voxels = self.R.shape[1]
            R_torch = torch.tensor(
                self.R, device=self.device, dtype=torch.float32
            )
            X_torch = torch.from_numpy(X).to(torch.float32).to(self.device)
            return (X_torch @ R_torch * n_voxels).cpu().numpy()
        elif self.backend == "geomloss":
            n_voxels = X.shape[1]
            X_torch_t = torch.tensor(
                np.ascontiguousarray(X.T),
                device=self.device,
                dtype=torch.float32,
            )
            return (self.R.T @ X_torch_t * n_voxels).T.cpu().numpy()


def _pot_solver(X_torch, Y_torch, evecs_torch, reg, alpha, max_iter):
    """Solve the OT problem using the POT library."""
    import ot

    M_func = ot.dist(X_torch, Y_torch)
    M_geom = ot.dist(evecs_torch)

    # Normalize both cost matrices to have the same scale
    M_func_normalized = ot.utils.cost_normalization(M_func, "max")
    M_geom_normalized = ot.utils.cost_normalization(M_geom, "max")
    M_normalized = (1 - alpha) * M_func_normalized + alpha * M_geom_normalized

    res = ot.solve(
        M=M_normalized,
        reg=reg,
        tol=1e-7,
        method="sinkhorn_log",
        max_iter=max_iter,
        verbose=False,
        grad="detach",
    )

    return res.plan.cpu().numpy()


def _geomloss_solver(X_torch, Y_torch, evecs_torch, reg, alpha, max_iter):
    """Solve the OT problem using the GeomLoss library."""
    import torch
    from geomloss import _backends as bk
    from geomloss._arguments import ArrayProperties
    from geomloss._typing import CostMatrices
    from geomloss.ot._abstract_solvers import (
        annealing_parameters,
        sinkhorn_loop,
    )
    from geomloss.ot._implementations.sample import (
        OTResultSample,
        cost_matrix,
        softmin_sample,
    )

    device = X_torch.device
    M_func = cost_matrix(X_torch, Y_torch, matrix_type="lazy")
    M_func_t = cost_matrix(Y_torch, X_torch, matrix_type="lazy")
    M_geom = cost_matrix(evecs_torch, evecs_torch, matrix_type="lazy")

    # Normalize both cost matrices to have the same scale
    M_func_normalized = M_func / M_func.max(axis=1).max()
    M_func_normalized_t = M_func_t / M_func.max(axis=1).max()
    M_geom_normalized = M_geom / M_geom.max(axis=1).max()
    M_normalized = (1 - alpha) * M_func_normalized + alpha * M_geom_normalized
    M_normalized_t = (
        1 - alpha
    ) * M_func_normalized_t + alpha * M_geom_normalized

    # Uniform weights for the source and target distributions
    N, M = M_normalized.shape
    a = torch.ones(N, dtype=torch.float32).to(device) / N
    b = torch.ones(M, dtype=torch.float32).to(device) / M

    array_properties = ArrayProperties(
        B=0,
        N=N,
        M=M,
        dtype=torch.float32,
        device=device,
        library="torch",
    )

    max_cost = M_normalized.max(axis=1).max()
    min_cost = M_normalized.min(axis=1).min()
    descent = annealing_parameters(
        maxmin_cost=max_cost - min_cost,
        eps=reg,
        rho=None,
        n_iter=max_iter,
    )
    costs_matrices = CostMatrices(
        xy=M_normalized, yx=M_normalized_t, xx=None, yy=None
    )

    potentials = sinkhorn_loop(
        softmin=softmin_sample,
        log_a_list=[bk.stable_log(a)],
        log_b_list=[bk.stable_log(b)],
        C_list=[costs_matrices],
        descent=descent,
        debias=False,
        last_extrapolation=True,
    )

    res = OTResultSample(
        X_a=None,
        X_b=None,
        a=a,
        b=b,
        C=costs_matrices,
        cost=None,
        reg=reg,
        reg_type="KL",
        debias=False,
        unbalanced=None,
        unbalanced_type="KL",
        potentials=potentials,
        array_properties=array_properties,
    )

    return res.plan_operator
