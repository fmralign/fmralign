from numpy.testing import assert_array_almost_equal
from sklearn.datasets import make_regression

from fmralign.methods.ridge import RidgeAlignment


def test_ridge_alignment():
    """Test RidgeAlignment with no regularization."""
    X, Y = make_regression()
    ridge_model = RidgeAlignment(alphas=[0])
    ridge_model.fit(X, Y)

    transformed_X = ridge_model.transform(X)
    assert_array_almost_equal(transformed_X, Y)
