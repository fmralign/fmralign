import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fmralign.alignment.utils import (
    _check_input_arrays,
    _check_labels,
    _check_method,
    _fit_template,
    _map_to_target,
    _rescaled_euclidean_mean,
)
from fmralign.methods import Identity
from fmralign.tests.utils import sample_subjects


@pytest.mark.parametrize("scale_average", [True, False])
def test_rescaled_euclidean_mean(scale_average):
    """Test the rescaled Euclidean mean function."""
    subjects_data, _ = sample_subjects()
    average_data = _rescaled_euclidean_mean(subjects_data)
    assert average_data.shape == subjects_data[0].shape
    assert average_data.dtype == subjects_data[0].dtype

    if scale_average is False:
        assert_array_equal(average_data, np.mean(subjects_data, axis=0))


def test_check_input_arrays():
    """Test the input array checking function."""
    # Valid input
    subjects_keys = ["sub-01", "sub-02", "sub-03"]
    subjects_values = [np.random.rand(10, 5) for _ in range(3)]
    subjects_dict = dict(zip(subjects_keys, subjects_values))
    checked_keys, checked_values = _check_input_arrays(subjects_dict)
    assert isinstance(checked_values, list)
    assert all(isinstance(x, np.ndarray) for x in checked_values)
    assert checked_keys == subjects_keys

    # Invalid input (not a dict)
    with pytest.raises(AttributeError):
        _check_input_arrays(np.random.rand(10, 5))

    # Invalid input (empty dict)
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        _check_input_arrays({})

    # Invalid input (non-array elements)
    invalid_dict = {"sub-01": np.random.rand(10, 5), "sub-02": "not an array"}
    with pytest.raises(
        ValueError, match="All elements in the input dict must be numpy arrays"
    ):
        _check_input_arrays(invalid_dict)

    # Invalid input (non-2D arrays)
    invalid_dict = {
        "sub-01": np.random.rand(10),
        "sub-02": np.random.rand(5, 1),
    }
    with pytest.raises(
        ValueError, match="All arrays in the input dict must be 2D arrays"
    ):
        _check_input_arrays(invalid_dict)

    # Invalid input (arrays with different number of features)
    invalid_dict = {
        "sub-01": np.random.rand(10, 5),
        "sub-02": np.random.rand(10, 6),
    }
    with pytest.raises(
        ValueError,
        match="All arrays in the input dict must have the same number of features",
    ):
        _check_input_arrays(invalid_dict)

    # Invalid input (arrays with different number of samples)
    invalid_dict = {
        "sub-01": np.random.rand(10, 5),
        "sub-02": np.random.rand(12, 5),
    }
    with pytest.raises(
        ValueError,
        match="All arrays in the input dict must have the same number of samples",
    ):
        _check_input_arrays(invalid_dict)


def test_check_labels():
    """Test the label checking function."""
    subjects_data, labels = sample_subjects()
    # Check valid labels
    _check_labels(subjects_data[0], labels)

    # Check invalid labels (length mismatch)
    with pytest.raises(ValueError):
        _check_labels(subjects_data[0], np.array([1, 2]))

    # Check invalid labels (not 1D)
    with pytest.raises(ValueError):
        _check_labels(subjects_data[0], np.array([[1, 2], [3, 4]]))

    # Check parcel size warning
    with pytest.warns(UserWarning):
        _check_labels(subjects_data[0], labels, threshold=2)

    # Check integer conversion warning
    with pytest.warns(UserWarning):
        _check_labels(subjects_data[0], labels.astype(float))

    # Check no labels warning
    with pytest.warns(UserWarning):
        _check_labels(subjects_data[0], None)


def test_check_method():
    """Test the method checking function."""
    # Check valid method
    method = _check_method("identity")
    assert isinstance(method, Identity)

    # Check invalid method
    with pytest.raises(ValueError):
        _check_method("invalid_method")

    # Check valid method instance
    method_instance = Identity()
    checked_method = _check_method(method_instance)
    assert checked_method is method_instance


def test_fit_template():
    """Test fitting a template to a set of subjects."""
    subjects_data, labels = sample_subjects()
    estimators, template = _fit_template(subjects_data, Identity(), labels)
    assert len(estimators) == len(subjects_data)
    euclidean_mean = _rescaled_euclidean_mean(subjects_data)
    # Check that the template is the Euclidean mean for identity method
    assert_array_equal(template, euclidean_mean)


def test_map_to_target():
    """Test identity of multiple subjects to target."""
    X, labels = sample_subjects()
    target_data = X[0]
    estimators = _map_to_target(X, target_data, Identity(), labels)
    assert len(estimators) == len(X)
    for estimator in estimators:
        assert isinstance(estimator, Identity)
        transformed_data = estimator.transform(X[0])
        assert transformed_data.shape == target_data.shape
        assert transformed_data.dtype == target_data.dtype
