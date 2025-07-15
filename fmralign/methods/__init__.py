from .identity import Identity
from .ot import OptimalTransportAlignment, SparseUOT
from .procrustes import ScaledOrthogonalAlignment
from .ridge import RidgeAlignment

__all__ = [
    "Identity",
    "OptimalTransportAlignment",
    "SparseUOT",
    "ScaledOrthogonalAlignment",
    "RidgeAlignment",
]
