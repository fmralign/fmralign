from fmralign.methods.utils import BaseMethod


class Identity(BaseMethod):
    """Compute no alignment, used as baseline for benchmarks : RX = X."""

    def transform(self, X):
        """Returns X"""
        return X
