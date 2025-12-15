import numpy as np
import ot
import torch

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

    Attributes
    ----------
    R : scipy.sparse.csr_matrix
        Transport plan computed during fitting.
    """

    def __init__(
        self,
        reg=1e-2,
        max_iter=1000,
        tol=1e-7,
        **kwargs,
    ):
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
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

        n = len(X.T)
        if n < 1000:
            res = ot.solve_sample(
                X.T,
                Y.T,
                reg=self.reg,
                tol=self.tol,
                max_iter=self.max_iter,
                lazy=False,
                **self.kwargs,
            )

            self.R = res.plan

        else:
            X_torch = torch.tensor(np.ascontiguousarray(X.T), device=DEVICE)
            Y_torch = torch.tensor(np.ascontiguousarray(Y.T), device=DEVICE)
            res = ot.solve_sample(
                X_torch,
                Y_torch,
                reg=self.reg,
                tol=self.tol,
                max_iter=self.max_iter,
                lazy=True,
                method="geomloss",
                **self.kwargs,
            )

            self.R = res.lazy_plan

    def transform(self, X):
        """Transform X using optimal coupling computed during fit."""
        n_voxels = X.shape[1]
        if isinstance(self.R, ot.utils.LazyTensor):
            X_torch = torch.tensor(X, device=DEVICE)
            X_aligned = self.lazy_matmul(X_torch, self.R).detach() * n_voxels
            return X_aligned.cpu().numpy()
        else:
            return X @ self.R * n_voxels

    @classmethod
    def lazy_matmul(self, A, B, batch_size=10000):
        """Internal function to perform lazy matrix multiplication.

        Parameters
        ----------
        A : array_like
            Left operand
        B : LazyTensor
            Right operand
        batch_size : int, optional, by default 10000

        Returns
        -------
        array_like
            The matrix product of A and B.
        """
        nx = ot.utils.get_backend(A[0:1], B[0:1])
        n1, n2 = A.shape
        n3 = B.shape[1]

        def getitem_AB(i, j, k, A, B):
            if isinstance(i, int):
                i = slice(i, i + 1)

            A_block = A[i, j]
            B_block = B[j, k]
            return nx.einsum("ij,jk->ijk", A_block, B_block)

        AB = ot.utils.LazyTensor((n1, n2, n3), getitem_AB, A=A, B=B)
        C = nx.zeros((n1, n3), type_as=A[0])
        for j in range(0, n3, batch_size):
            C[:, j : j + batch_size] = nx.sum(
                AB[:, :, j : j + batch_size], axis=1
            )

        return C
