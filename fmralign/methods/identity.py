from fmralign.methods.base import BaseAlignment


class Identity(BaseAlignment):
    """Compute no alignment, used as baseline for benchmarks : RX = X."""

    def transform(self, X):
        """Returns X"""
        return X
