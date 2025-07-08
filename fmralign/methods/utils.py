from sklearn.base import BaseEstimator, TransformerMixin


class BaseMethod(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def transform(self, X):
        pass
