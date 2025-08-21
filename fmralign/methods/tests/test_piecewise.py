import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fmralign.methods import (
    Identity,
    OptimalTransport,
    Procrustes,
    RidgeAlignment,
    SparseUOT,
)
from fmralign.methods.piecewise import (
    PiecewiseAlignment,
    _array_to_list,
    _fit_one_piece,
    _list_to_array,
    _transform_one_piece,
)
from fmralign.tests.utils import sample_labels, sample_one_subject

methods = [
    Identity(),
    Procrustes(),
    SparseUOT(),
    RidgeAlignment(),
    OptimalTransport(),
]


def test_fit_one_piece():
    """Test fitting a single piece of data."""
    X = np.random.rand(10, 5)
    Y = np.random.rand(10, 5)
    method = Identity()
    fitted_piece = _fit_one_piece(X, Y, method)
    assert hasattr(fitted_piece, "fit")


def test_transform_one_piece():
    """Test transforming a single piece of data."""
    X = np.random.rand(10, 5)
    method = Identity()
    fitted_piece = _fit_one_piece(X, X, method)
    transformed_piece = _transform_one_piece(X, fitted_piece)
    assert_array_equal(transformed_piece, X)


def test_array_to_list():
    """Test converting an array to a list based on labels."""
    arr = np.random.rand(10, 5)
    labels = np.array([0, 1, 0, 1, 2])
    result = _array_to_list(arr, labels)
    assert len(result) == 3
    assert result[0].shape == (10, 2)  # Two columns for label 0
    assert result[1].shape == (10, 2)  # Two columns for label 1
    assert result[2].shape == (10, 1)  # One column for label 2


def test_list_to_array():
    """Test converting a list back to an array based on labels."""
    lst = [np.random.rand(10, 2), np.random.rand(10, 2), np.random.rand(10, 1)]
    labels = np.array([0, 1, 0, 1, 2])
    result = _list_to_array(lst, labels)
    assert result.shape == (10, 5)  # Should match original shape
    assert_array_equal(result[:, labels == 0], lst[0])
    assert_array_equal(result[:, labels == 1], lst[1])
    assert_array_equal(result[:, labels == 2], lst[2])


@pytest.mark.parametrize("method", methods)
def test_piecewise_all_methods(method):
    """Test PiecewiseAlignment with all methods."""
    X = sample_one_subject(n_features=10, n_voxels=30)
    Y = sample_one_subject(n_features=10, n_voxels=30)
    labels = sample_labels(n_voxels=30, n_labels=3)
    algo = PiecewiseAlignment(
        method=method,
        labels=labels,
    )
    algo.fit(X, Y)
    transformed = algo.transform(X)
    assert transformed.shape == X.shape


test_piecewise_all_methods(RidgeAlignment())
