# -*- coding: utf-8 -*-
"""Module implementing alignment estimators on ndarrays."""

import warnings

import numpy as np
import ot
import torch
from fugw.solvers.utils import (
    batch_elementwise_prod_and_sum,
    crow_indices_to_row_indices,
    solver_sinkhorn_sparse,
)
from fugw.utils import _low_rank_squared_l2, _make_csr_matrix
from joblib import Parallel, delayed
from scipy import linalg
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV


def scaled_procrustes(X, Y, scaling=False, primal=None):
    """
    Compute a mixing matrix R and a scaling sc such that Frobenius norm
    ||sc RX - Y||^2 is minimized and R is an orthogonal matrix

    Parameters
    ----------
    X: (n_samples, n_features) nd array
        source data
    Y: (n_samples, n_features) nd array
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
    ----------
    R: (n_features, n_features) nd array
        transformation matrix
    sc: int
        scaling parameter
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    if np.linalg.norm(X) == 0 or np.linalg.norm(Y) == 0:
        return np.eye(X.shape[1]), 1
    if primal is None:
        primal = X.shape[0] >= X.shape[1]
    if primal:
        A = Y.T.dot(X)
        if A.shape[0] == A.shape[1]:
            A += +1.0e-18 * np.eye(A.shape[0])
        U, s, V = linalg.svd(A, full_matrices=0)
        R = U.dot(V)
    else:  # "dual" mode
        Uy, sy, Vy = linalg.svd(Y, full_matrices=0)
        Ux, sx, Vx = linalg.svd(X, full_matrices=0)
        A = np.diag(sy).dot(Uy.T).dot(Ux).dot(np.diag(sx))
        U, s, V = linalg.svd(A)
        R = Vy.T.dot(U).dot(V).dot(Vx)

    if scaling:
        sc = s.sum() / (np.linalg.norm(X) ** 2)
    else:
        sc = 1
    return R.T, sc


def _projection(x, y):
    """
    Compute scalar d minimizing ||dx-y||.

    Parameters
    ----------
    x: (n_features) nd array
        source vector
    y: (n_features) nd array
        target vector

    Returns
    --------
    d: int
        scaling factor
    """
    if (x == 0).all():
        return 0
    else:
        return np.dot(x, y) / np.linalg.norm(x) ** 2


def _voxelwise_signal_projection(X, Y, n_jobs=1, parallel_backend="threading"):
    """
    Compute D, list of scalar d_i minimizing :
    ||d_i * x_i - y_i|| for every x_i, y_i in X, Y

    Parameters
    ----------
    X: (n_samples, n_features) nd array
        source data
    Y: (n_samples, n_features) nd array
        target data

    Returns
    --------
    D: list of ints
        List of optimal scaling factors
    """
    return Parallel(n_jobs, parallel_backend)(
        delayed(_projection)(voxel_source, voxel_target)
        for voxel_source, voxel_target in zip(X, Y)
    )


class Alignment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def transform(self, X):
        pass


class Identity(Alignment):
    """Compute no alignment, used as baseline for benchmarks : RX = X."""

    def transform(self, X):
        """Returns X"""
        return X


class DiagonalAlignment(Alignment):
    """
    Compute the voxelwise projection factor between X and Y.

    Parameters
    ----------
    n_jobs: integer, optional (default = 1)
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    parallel_backend: str, ParallelBackendBase instance, None (default: 'threading')
        Specify the parallelization backend implementation. For more
        informations see joblib.Parallel documentation

    Attributes
    -----------
    R : scipy.sparse.diags
        Scaling matrix containing the optimal shrinking factor for every voxel
    """

    def __init__(self, n_jobs=1, parallel_backend="threading"):
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    def fit(self, X, Y):
        """

        Parameters
        --------------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        shrinkage_coefficients = _voxelwise_signal_projection(
            X.T, Y.T, self.n_jobs, self.parallel_backend
        )

        self.R = diags(shrinkage_coefficients)
        return

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return self.R.dot(X.T).T


class ScaledOrthogonalAlignment(Alignment):
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
    """

    def __init__(self, scaling=True):
        self.scaling = scaling
        self.scale = 1

    def fit(self, X, Y):
        """
        Fit orthogonal R s.t. ||sc XR - Y||^2

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        R, sc = scaled_procrustes(X, Y, scaling=self.scaling)
        self.scale = sc
        self.R = sc * R
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return X.dot(self.R)


class RidgeAlignment(Alignment):
    """
    Compute a scikit-estimator R using a mixing matrix M s.t Frobenius
    norm || XM - Y ||^2 + alpha * ||M||^2 is minimized with cross-validation

    Parameters
    ----------
    R : scikit-estimator from sklearn.linear_model.RidgeCV
        with methods fit, predict
    alpha : numpy array of shape [n_alphas]
        Array of alpha values to try. Regularization strength;
        must be a positive float. Regularization improves the conditioning
        of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization. Alpha corresponds to
        ``C^-1`` in other models such as LogisticRegression or LinearSVC.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        -None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
    """

    def __init__(self, alphas=[0.1, 1.0, 10.0, 100, 1000], cv=4):
        self.alphas = [alpha for alpha in alphas]
        self.cv = cv

    def fit(self, X, Y):
        """
        Fit R s.t. || XR - Y ||^2 + alpha ||R||^2 is minimized with cv

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        self.R = RidgeCV(
            alphas=self.alphas,
            fit_intercept=True,
            scoring="r2",
            cv=self.cv,
        )
        self.R.fit(X, Y)
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return self.R.predict(X)


class OptimalTransportAlignment(Alignment):
    """
    Compute the optimal coupling between X and Y with entropic regularization,
    using the pure Python POT (https://pythonot.github.io/) package.

    Parameters
    ----------
    solver : str (optional)
        solver from POT called to find optimal coupling 'sinkhorn',
        'greenkhorn', 'sinkhorn_stabilized','sinkhorn_epsilon_scaling', 'exact'
        see POT/ot/bregman on Github for source code of solvers
    metric : str (optional)
        metric used to create transport cost matrix,
        see full list in scipy.spatial.distance.cdist doc
    reg : int (optional)
        level of entropic regularization

    Attributes
    ----------
    R : scipy.sparse.csr_matrix
        Mixing matrix containing the optimal permutation
    """

    def __init__(
        self,
        solver="sinkhorn_epsilon_scaling",
        metric="euclidean",
        reg=1,
        max_iter=1000,
        tol=1e-3,
    ):
        self.solver = solver
        self.metric = metric
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        """

        Parameters
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """

        n = len(X.T)
        if n > 5000:
            warnings.warn(
                f"One parcel is {n} voxels. As optimal transport on this region "
                "would take too much time, no alignment was performed on it. "
                "Decrease parcel size to have intended behavior of alignment."
            )
            self.R = np.eye(n)
            return self
        else:
            a = np.ones(n) * 1 / n
            b = np.ones(n) * 1 / n

            M = cdist(X.T, Y.T, metric=self.metric)

            if self.solver == "exact":
                self.R = ot.lp.emd(a, b, M) * n
            else:
                self.R = (
                    ot.sinkhorn(
                        a,
                        b,
                        M,
                        self.reg,
                        method=self.solver,
                        numItermax=self.max_iter,
                        stopThr=self.tol,
                    )
                    * n
                )
            return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        return X.dot(self.R)


class SparseUOT(Alignment):
    """
    Compute the unbalanced regularized optimal coupling between X and Y,
    with sparsity constraints inspired by the FUGW package sparse
    sinkhorn solver.
    (https://github.com/alexisthual/fugw/blob/main/src/fugw/solvers/sparse.py)

    Parameters
    ----------
    sparsity_mask : sparse torch.Tensor of shape (n_features, n_features)
        Sparse mask that defines the sparsity pattern of the coupling matrix.
    rho : float (optional)
        Strength of the unbalancing constraint. Lower values will favor lower
        mass transport. Defaults to infinity.
    reg : float (optional)
        Strength of the entropic regularization. Defaults to 0.1.
    max_iter : int (optional)
        Maximum number of iterations. Defaults to 1000.
    tol : float (optional)
        Tolerance for stopping criterion. Defaults to 1e-7.
    eval_freq : int (optional)
        Frequency of evaluation of the stopping criterion. Defaults to 10.
    device : str (optional)
        Device on which to perform computations. Defaults to 'cpu'.
    verbose : bool (optional)
        Whether to print progress information. Defaults to False.

    Attributes
    ----------
    pi : sparse torch.Tensor of shape (n_features, n_features)
        Sparse coupling matrix
    """

    def __init__(
        self,
        sparsity_mask,
        rho=float("inf"),
        reg=1,
        max_iter=1000,
        tol=1e-3,
        eval_freq=10,
        device="cpu",
        verbose=False,
    ):
        self.rho = rho
        self.reg = reg
        self.sparsity_mask = sparsity_mask
        self.max_iter = max_iter
        self.tol = tol
        self.eval_freq = eval_freq
        self.device = device
        self.verbose = verbose

    def _initialize_weights(self, n, cost):
        crow_indices, col_indices = cost.crow_indices(), cost.col_indices()
        row_indices = crow_indices_to_row_indices(crow_indices)
        weights = torch.ones(n, device=self.device) / n
        ws_dot_wt_values = weights[row_indices] * weights[col_indices]
        ws_dot_wt = _make_csr_matrix(
            crow_indices,
            col_indices,
            ws_dot_wt_values,
            cost.size(),
            self.device,
        )
        return weights, ws_dot_wt

    def _initialize_plan(self, n):
        return (
            torch.sparse_coo_tensor(
                self.sparsity_mask.indices(),
                torch.ones_like(self.sparsity_mask.values())
                / self.sparsity_mask.values().shape[0],
                (n, n),
            )
            .coalesce()
            .to_sparse_csr()
            .to(self.device)
        )

    def _uot_cost(self, init_plan, F, n):
        crow_indices, col_indices = (
            init_plan.crow_indices(),
            init_plan.col_indices(),
        )
        row_indices = crow_indices_to_row_indices(crow_indices)
        cost_values = batch_elementwise_prod_and_sum(
            F[0], F[1], row_indices, col_indices, 1
        )
        # Clamp negative values to avoid numerical errors
        cost_values = torch.clamp(cost_values, min=0.0)
        cost_values = torch.sqrt(cost_values)
        return _make_csr_matrix(
            crow_indices,
            col_indices,
            cost_values,
            (n, n),
            self.device,
        )

    def fit(self, X, Y):
        """

        Parameters
        ----------
        X: (n_samples, n_features) torch.Tensor
            source data
        Y: (n_samples, n_features) torch.Tensor
            target data
        """
        n_features = X.shape[1]
        F = _low_rank_squared_l2(X.T, Y.T)

        init_plan = self._initialize_plan(n_features)
        cost = self._uot_cost(init_plan, F, n_features)

        weights, ws_dot_wt = self._initialize_weights(n_features, cost)

        uot_params = (
            torch.tensor([self.rho], device=self.device),
            torch.tensor([self.rho], device=self.device),
            torch.tensor([self.reg], device=self.device),
        )
        init_duals = (
            torch.zeros(n_features, device=self.device),
            torch.zeros(n_features, device=self.device),
        )
        tuple_weights = (weights, weights, ws_dot_wt)
        train_params = (self.max_iter, self.tol, self.eval_freq)

        _, pi = solver_sinkhorn_sparse(
            cost=cost,
            init_duals=init_duals,
            uot_params=uot_params,
            tuple_weights=tuple_weights,
            train_params=train_params,
            verbose=self.verbose,
        )

        # Convert pi to coo format
        self.R = pi.to_sparse_coo().detach() * n_features

        if self.R.values().isnan().any():
            raise ValueError(
                "Coupling matrix contains NaN values,"
                "try increasing the regularization parameter."
            )

        return self

    def transform(self, X):
        """Transform X using optimal coupling computed during fit.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
            Input data to be transformed

        Returns
        -------
        torch.Tensor of shape (n_samples, n_features)
            Transformed data
        """
        return (X @ self.R).to_dense()
