import numpy as np
from numpy.testing import assert_array_almost_equal
from fmralign.methods.identity import Identity


def test_identity_transform():
    """Test that Identity method returns the input unchanged."""
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)

    identity_method = Identity()
    transformed_X = identity_method.transform(X)

    assert_array_almost_equal(transformed_X, X)
