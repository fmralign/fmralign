import numpy as np
from numpy.testing import assert_array_almost_equal

from fmralign.methods.optimal_transport import OptimalTransport


def test_identity_wasserstein():
    """Test that the optimal coupling matrix is the\n
    identity matrix when using the identity alignment."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    algo = OptimalTransport(reg=1e-12)
    algo.fit(X, X)
    # Check if transport matrix P is uniform diagonal
    assert_array_almost_equal(algo.R, np.eye(n_features) / n_features)
    # Check if transformation preserves input
    assert_array_almost_equal(X, algo.transform(X))


def test_regularization_effect():
    """Test the effect of regularization parameter."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Compare results with different regularization values
    algo1 = OptimalTransport(reg=1.0)
    algo2 = OptimalTransport(reg=1e-3)

    algo1.fit(X, Y)
    algo2.fit(X, Y)

    # Higher regularization should lead to more uniform transport matrix
    assert np.std(algo1.R) < np.std(algo2.R)


def test_geomloss_backend():
    """Test the geomloss backend for large number of voxels."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)

    algo = OptimalTransport(reg=1e-12, backend="geomloss")
    algo.fit(X, X)
    X_transformed = algo.transform(X)

    # Check that we recover the identity mapping
    assert algo.R.shape == (n_features, n_features)
    assert isinstance(X_transformed, np.ndarray)
    assert_array_almost_equal(X_transformed, X)
