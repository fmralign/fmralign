import numpy as np
import pytest
from numpy.testing import assert_array_equal

from fmralign import GroupAlignment
from fmralign.alignment.utils import _check_method
from fmralign.tests.utils import sample_subjects

methods = ["identity", "ot", "sparse_uot", "procrustes", "ridge", "srm"]


@pytest.mark.parametrize("method", methods)
def test_alignment_template(method):
    """Test template alignment."""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    algo = GroupAlignment(method=method, labels=labels)
    algo.fit(X, y="template")

    assert isinstance(algo.method_, _check_method(method).__class__)
    assert len(algo.fitted_estimators) == len(X)
    if method != "srm":
        assert algo.template.shape == X[0].shape
    for i, x in X.items():
        transformed = algo._transform_one_array(x, algo.fitted_estimators[i])
        if method == "identity":
            assert_array_equal(transformed, x)
        elif method == "srm":
            n_components = algo.fitted_estimators[0].Wt.shape[1]
            assert transformed.shape == (x.shape[0], n_components)
        else:
            assert transformed.shape == x.shape


@pytest.mark.parametrize("method", methods)
def test_alignment_target(method):
    """Test alignment to a target"""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    target = X[0]
    algo = GroupAlignment(method=method, labels=labels)
    algo.fit(X, y=target)

    assert isinstance(algo.method_, _check_method(method).__class__)
    assert len(algo.fitted_estimators) == len(X)
    assert algo.template is None
    for i, x in X.items():
        transformed = algo._transform_one_array(x, algo.fitted_estimators[i])
        if method == "identity":
            assert_array_equal(transformed, x)
        elif method == "srm":
            n_components = algo.fitted_estimators[0].Wt.shape[1]
            assert transformed.shape == (x.shape[0], n_components)
        else:
            assert transformed.shape == x.shape


@pytest.mark.parametrize("method", methods)
def test_transform(method):
    """Test transform method."""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    algo = GroupAlignment(method=method, labels=labels)
    algo.fit(X)

    transformed_arrays = algo.transform(X)
    for i, x in X.items():
        if method == "identity":
            assert_array_equal(transformed_arrays[i], x)
        elif method == "srm":
            n_components = algo.fitted_estimators[0].Wt.shape[1]
            assert transformed_arrays[i].shape == (x.shape[0], n_components)
        else:
            assert transformed_arrays[i].shape == x.shape


@pytest.mark.parametrize("method", methods)
def test_predict_subject(method):
    """Test prediction of a new subject using the template."""
    data_train, labels = sample_subjects(
        n_subjects=3, n_features=10, n_voxels=40
    )
    data_test, _ = sample_subjects(n_subjects=3, n_features=15, n_voxels=40)

    X_train = dict(enumerate(data_train[:2]))
    X_test = dict(enumerate(data_test[:2]))
    Y_train, Y_test = data_train[2], data_test[2]

    algo = GroupAlignment(method=method, labels=labels)
    algo.fit(X_train)
    Y_pred = algo.predict_subject(X_test, Y_train)

    if method == "identity":
        assert_array_equal(Y_pred, np.mean(list(X_test.values()), axis=0))
    else:
        assert Y_pred.shape == Y_test.shape
