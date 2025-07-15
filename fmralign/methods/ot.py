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
from scipy.spatial.distance import cdist
from fmralign.methods.base import BaseAlignment


class OptimalTransport(BaseAlignment):
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


class SparseUOT(BaseAlignment):
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
        sparsity_mask=None,
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
        if self.sparsity_mask is None:
            # If no sparsity mask is provided, use a dense mask
            self.sparsity_mask = torch.ones(
                (n_features, n_features), device=self.device
            ).to_sparse_coo()
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
        X_ = torch.tensor(X, dtype=torch.float32).to(self.device)
        return (X_ @ self.R).to_dense().detach().cpu().numpy()
