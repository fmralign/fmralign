import numpy as np
from sklearn.linear_model import RidgeCV

from fmralign.methods.base import BaseAlignment


class RidgeAlignment(BaseAlignment):
    r"""
    Batched version. Compute a scikit-estimator R using a mixing matrix M s.t Frobenius
    norm :math:`|| XM - Y ||^2 + alpha * ||M||^2` is minimized, independently
    for each element of the batch dimension.
    cross-validation is used to find the optimal alpha.
    Parameters
    ----------
    R : list of scikit-estimators from sklearn.linear_model.RidgeCV
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

    def __init__(self, alphas=None, cv=4):
        self.alphas = alphas
        self.cv = cv

    def fit(self, X, Y):
        r"""
        Fit R s.t. :math:`|| XR - Y ||^2 + alpha ||R||^2` is minimized with cv,
        independently for each batch element.
        Parameters
        ----------
        X: (b, n_samples, n_features) nd array
            source data
        Y: (b, n_samples, n_features) nd array
            target data
        """
        b = X.shape[0]
        self.R = []
        for i in range(b):
            r = RidgeCV(
                alphas=(
                    self.alphas
                    if self.alphas is not None
                    else [0.1, 1.0, 10.0, 100, 1000]
                ),
                fit_intercept=True,
                scoring="r2",
                cv=self.cv,
            )
            r.fit(X[i], Y[i])
            self.R.append(r)
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit."""
        return np.stack(
            [r.predict(X[i]) for i, r in enumerate(self.R)], axis=0
        )
