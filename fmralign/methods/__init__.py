from .identity import Identity
from .ot import OptimalTransport, SparseUOT
from .procrustes import ScaledOrthogonal
from .ridge import RidgeAlignment

__all__ = [
    "Identity",
    "OptimalTransport",
    "SparseUOT",
    "ScaledOrthogonal",
    "RidgeAlignment",
]
