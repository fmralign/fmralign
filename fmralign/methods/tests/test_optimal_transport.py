import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

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
