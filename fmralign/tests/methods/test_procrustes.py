import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import orthogonal_procrustes

from fmralign.methods.procrustes import (
    ScaledOrthogonal,
    scaled_procrustes,
)


def test_scaled_procrustes_algorithmic():
    """Test Scaled procrustes"""
    X = np.random.randn(10, 20)
    Y = np.zeros_like(X)
    R = np.eye(X.shape[1])
    R_test, _ = scaled_procrustes(X, Y)
    assert_array_almost_equal(R, R_test)

    """Test if scaled_procrustes basis is orthogonal"""
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)

    Y = np.random.rand(3, 4)
    Y = Y - Y.mean(axis=1, keepdims=True)

    R, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R.dot(R.T), np.eye(R.shape[0]))
    assert_array_almost_equal(R.T.dot(R), np.eye(R.shape[0]))

    """ Test if it sticks to scipy scaled procrustes in a simple case"""
    X = np.random.rand(4, 4)
    Y = np.random.rand(4, 4)

    R, _ = scaled_procrustes(X, Y)
    R_s, _ = orthogonal_procrustes(Y, X)
    assert_array_almost_equal(R.T, R_s)

    """Test that primal and dual give same results"""
    # number of samples n , number of voxels p
    n, p = 100, 20
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R_s, _ = orthogonal_procrustes(Y, X)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(R1, R2)
    assert_array_almost_equal(R2, R_s.T)
    n, p = 20, 100
    X = np.random.randn(n, p)
    Y = np.random.randn(n, p)
    R1, s1 = scaled_procrustes(X, Y, scaling=True, primal=True)
    R_s, _ = orthogonal_procrustes(Y, X)
    R2, s2 = scaled_procrustes(X, Y, scaling=True, primal=False)
    assert_array_almost_equal(s1 * X.dot(R1), s2 * X.dot(R2))


def test_scaled_procrustes_on_simple_exact_cases():
    """Orthogonal Matrix"""
    v = 10
    k = 10
    rnd_matrix = np.random.rand(v, k)
    R, _ = np.linalg.qr(rnd_matrix)
    X = np.random.rand(10, 20)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)
    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(R_test.T, R)

    """Scaled Matrix"""
    X = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 4.0, 6.0], [7.0, 8.0, -5.0, -2.0]]
    )

    X = X - X.mean(axis=1, keepdims=True)

    Y = 2 * X
    Y = Y - Y.mean(axis=1, keepdims=True)

    assert_array_almost_equal(
        scaled_procrustes(X.T, Y.T, scaling=True)[0], np.eye(3)
    )
    assert_array_almost_equal(scaled_procrustes(X.T, Y.T, scaling=True)[1], 2)

    """3D Rotation"""
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(1), -np.sin(1)],
            [0.0, np.sin(1), np.cos(1)],
        ]
    )
    X = np.random.rand(3, 4)
    X = X - X.mean(axis=1, keepdims=True)
    Y = R.dot(X)

    R_test, _ = scaled_procrustes(X.T, Y.T)
    assert_array_almost_equal(
        R.dot(np.array([0.0, 1.0, 0.0])), np.array([0.0, np.cos(1), np.sin(1)])
    )
    assert_array_almost_equal(
        R.dot(np.array([0.0, 0.0, 1.0])),
        np.array([0.0, -np.sin(1), np.cos(1)]),
    )
    assert_array_almost_equal(R, R_test.T)

    """Test Scaled_Orthogonal_Alignment on an exact case"""
    ortho_al = ScaledOrthogonal(scaling=False)
    ortho_al.fit(X.T, Y.T)
    assert_array_almost_equal(ortho_al.transform(X.T), Y.T)
