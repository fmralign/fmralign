from sklearn.base import BaseEstimator, TransformerMixin
from fmralign.alignment.utils import (
    _check_labels,
    _check_method,
    _fit_template,
    _map_to_target,
)
import numpy as np


class GroupAlignment(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        method="identity",
        target=None,
        labels=None,
        n_jobs=1,
        verbose=0,
        n_iter=2,
        scale_template=False,
    ):
        self.method = method
        self.target = target
        self.labels = labels
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_iter = n_iter
        self.scale_template = scale_template
        self.template = None

    def fit(self, X, y=None) -> None:
        self.labels_ = _check_labels(X[0], self.labels, verbose=self.verbose)
        self.method_ = _check_method(self.method, self.labels_)

        if self.target is None:  # Template alignment
            self.fit_, self.template = _fit_template(
                X,
                self.method,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
                self.n_iter,
                self.scale_template,
            )
        elif isinstance(self.target, np.ndarray):  # Pairwise alignment
            self.fit_ = _map_to_target(
                X,
                self.target,
                self.method,
                self.labels_,
                self.n_jobs,
                max(self.verbose - 1, 0),
            )
        else:
            raise ValueError(
                "Target must be an integer index of the subject "
                "or None for template alignment."
            )

    def _transform_one_array(self, X, estimator):
        # Check if estimator is fitted
        if not hasattr(estimator, "fit"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )
        return estimator.transform(X)

    def transform(self, X, subject_indices):
        return [
            self._transform_one_array(X[i], self.fit_[i])
            for i in subject_indices
        ]

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here.

        Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no 'fit_transform' method"
        )
