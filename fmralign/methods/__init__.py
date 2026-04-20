from .identity import Identity
from .optimal_transport import OptimalTransport, SpectralOT
from .procrustes import Procrustes
from .ridge import RidgeAlignment
from .srm import DetSRM

__all__ = [
    "DetSRM",
    "Identity",
    "OptimalTransport",
    "Procrustes",
    "RidgeAlignment",
    "SpectralOT",
]
