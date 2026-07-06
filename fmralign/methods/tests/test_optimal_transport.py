import itertools
import re

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from fmralign.methods import optimal_transport
from fmralign.methods.optimal_transport import OptimalTransport, SpectralOT


def test_identity_wasserstein():
    """Test that the optimal coupling matrix is the\n
    identity matrix when using the identity alignment."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    algo = OptimalTransport(reg=1e-6)
    algo.fit(X, X)
    # Check if transport matrix P is uniform diagonal
    assert_array_almost_equal(algo.R, np.eye(n_features) / n_features)
    # Check if transformation preserves input
    assert_array_almost_equal(X, algo.transform(X))


def test_regularization_effect():
    """Test the effect of regularization parameter."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)

    # Compare results with different regularization values
    algo1 = OptimalTransport(reg=10.0)
    algo2 = OptimalTransport(reg=1e-3)

    algo1.fit(X, X)
    algo2.fit(X, X)

    # Higher regularization should lead to flatter diagonal
    assert np.all(np.diag(algo1.R) < np.diag(algo2.R))


@pytest.mark.parametrize("backend", ["pot", "geomloss"])
def test_spectral_ot(backend):
    """Test SpectralOT against OptimalTransport on synthetic data."""
    n_samples, n_features = 10, 5
    X_train = np.random.randn(n_samples, n_features)
    Y_train = X_train + 0.1 * np.random.randn(n_samples, n_features)
    X_test = np.random.randn(n_samples, n_features)

    evecs = np.random.randn(3, n_features)

    # Check the anatomical case (alpha=1.0)
    algo_ot = OptimalTransport(reg=1e-3)
    algo_spectral_anat = SpectralOT(
        reg=1e-3, evecs=evecs, alpha=1.0, backend=backend
    )
    algo_ot.fit(evecs, evecs)
    algo_spectral_anat.fit(X_train, Y_train)
    assert_array_almost_equal(
        algo_ot.transform(X_test),
        algo_spectral_anat.transform(X_test),
        decimal=5,
    )

    # Check the functional case (alpha=0.0)
    algo_ot.fit(X_train, Y_train)
    algo_spectral_func = SpectralOT(
        reg=1e-3, evecs=evecs, alpha=0.0, backend=backend
    )
    algo_spectral_func.fit(X_train, Y_train)
    assert_array_almost_equal(
        algo_ot.transform(X_test),
        algo_spectral_func.transform(X_test),
        decimal=5,
    )


@pytest.mark.parametrize(
    "has_torch,has_pot,has_geomloss",
    list(itertools.product([True, False], repeat=3)),
)
def test_dependency_guards_all_combinations(
    monkeypatch, has_torch, has_pot, has_geomloss
):
    """Test that OptimalTransport and SpectralOT raise\n
    ImportError when dependencies are missing."""
    monkeypatch.setattr(optimal_transport, "_HAS_TORCH", has_torch)
    monkeypatch.setattr(optimal_transport, "_HAS_POT", has_pot)
    monkeypatch.setattr(optimal_transport, "_HAS_GEOMLOSS", has_geomloss)

    evecs = np.random.randn(3, 5)
    pot_ok = has_torch and has_pot
    geomloss_ok = has_torch and has_geomloss

    # OptimalTransport needs torch + POT
    if pot_ok:
        OptimalTransport(max_iter=1)
    else:
        with pytest.raises(
            ImportError, match=re.escape("pip install fmralign[ot_pot]")
        ):
            OptimalTransport(max_iter=1)

    # SpectralOT(backend="pot") needs torch + POT
    if pot_ok:
        SpectralOT(evecs, backend="pot", max_iter=1)
    else:
        with pytest.raises(
            ImportError, match=re.escape("pip install fmralign[ot_pot]")
        ):
            SpectralOT(evecs, backend="pot", max_iter=1)

    # SpectralOT(backend="geomloss") needs torch + GeomLoss
    if geomloss_ok:
        SpectralOT(evecs, backend="geomloss", max_iter=1)
    else:
        with pytest.raises(
            ImportError, match=re.escape("pip install fmralign[ot_geomloss]")
        ):
            SpectralOT(evecs, backend="geomloss", max_iter=1)
