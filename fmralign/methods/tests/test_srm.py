import numpy as np
from numpy.testing import assert_array_almost_equal

from fmralign.methods import DetSRM


def test_identity():
    """Test fitting on identical source/target"""
    n_samples, n_components = 10, 4
    X = np.random.rand(n_samples, n_components)

    srm = DetSRM(n_components=n_components)
    srm.fit(X, X)
    assert_array_almost_equal(srm.W, np.eye(n_components))
    assert_array_almost_equal(srm.transform(X), X)


def test_basis_orthogonality():
    """Test that the basis W is orthogonal in the components"""
    n_samples, n_voxels, n_components = 10, 8, 4
    X = np.random.rand(n_samples, n_voxels)
    S = np.random.rand(n_samples, n_components)
    srm = DetSRM(n_components).fit(X, S)
    W = srm.W
    assert_array_almost_equal(W @ W.T, np.eye(n_components))
