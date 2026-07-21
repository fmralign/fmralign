from sklearn.base import BaseEstimator, TransformerMixin


class BaseAlignment(TransformerMixin, BaseEstimator):
    """Base class for all alignment methods."""

    def __init__(self):
        pass

    def fit(self, X, Y):
        return self

    def transform(self, X):
        pass
