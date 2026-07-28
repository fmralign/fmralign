from .identity import Identity
from .optimal_transport import OptimalTransport, SpectralOT
from .piecewise import PiecewiseAlignment
from .procrustes import Procrustes
from .ridge import RidgeAlignment
from .srm import DetSRM

__all__ = [
    "DetSRM",
    "Identity",
    "OptimalTransport",
    "PiecewiseAlignment",
    "Procrustes",
    "RidgeAlignment",
    "SpectralOT",
]
