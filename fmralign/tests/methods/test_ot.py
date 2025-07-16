import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from fmralign.methods.ot import (
    OptimalTransport,
    SparseUOT,
)


def test_ot_backend():
    n_samples, n_features = 100, 20
    epsilon = 1e-2
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)
    X /= np.linalg.norm(X)
    Y /= np.linalg.norm(Y)
    pot_algo = OptimalTransport(reg=epsilon)
    sparsity_mask = torch.ones(n_features, n_features).to_sparse_coo()
    torch_algo = SparseUOT(sparsity_mask=sparsity_mask, reg=epsilon)
    pot_algo.fit(X, Y)
    torch_algo.fit(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    assert_array_almost_equal(
        pot_algo.R, torch_algo.R.to_dense().numpy(), decimal=3
    )


def test_identity_wasserstein():
    """Test that the optimal coupling matrix is the\n
    identity matrix when using the identity alignment."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    algo = OptimalTransport(reg=1e-12)
    algo.fit(X, X)
    # Check if transport matrix P is uniform diagonal
    assert_array_almost_equal(algo.R, np.eye(n_features))
    # Check if transformation preserves input
    assert_array_almost_equal(X, algo.transform(X))


def test_regularization_effect():
    """Test the effect of regularization parameter."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_features)

    # Compare results with different regularization values
    algo1 = OptimalTransport(reg=1e-1)
    algo2 = OptimalTransport(reg=1e-3)

    algo1.fit(X, Y)
    algo2.fit(X, Y)

    # Higher regularization should lead to more uniform transport matrix
    assert np.std(algo1.R) < np.std(algo2.R)


def test_sparseuot():
    """Test the sparse version of optimal transport."""
    n_samples, n_features = 100, 20
    X = torch.randn(n_samples, n_features)
    Y = torch.randn(n_samples, n_features)
    X /= torch.norm(X)
    Y /= torch.norm(Y)
    algo = SparseUOT()
    algo.fit(X, Y)
    X_transformed = algo.transform(X)

    assert algo.R.shape == (n_features, n_features)
    assert algo.R.dtype == torch.float32
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape == X.shape

    # Test identity transformation
    algo.R = (torch.eye(n_features)).to_sparse_coo()
    X_transformed = algo.transform(X)
    assert_array_almost_equal(X_transformed, X)

    # Check the unbalanced case
    algo = SparseUOT()
    algo.fit(X, Y)
    mass1 = algo.R.sum() / n_features

    algo = SparseUOT(rho=0.1)
    algo.fit(X, Y)
    mass2 = algo.R.sum() / n_features

    algo = SparseUOT(rho=0.0)
    algo.fit(X, Y)
    mass3 = algo.R.sum() / n_features

    assert torch.allclose(mass1, torch.tensor(1.0))
    assert mass1 > mass2 > mass3
