from fmralign.methods.base import BaseAlignment


class Identity(BaseAlignment):
    """Compute no alignment, used as baseline for benchmarks : RX = X."""

    def fit(self, X, Y):
        raise NotImplementedError(
            "Identity method does not require fitting. Use transform directly."
        )

    def transform(self, X):
        """Returns X"""
        return X
