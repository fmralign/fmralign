from numpy.testing import assert_array_equal

from fmralign import GroupAlignment
from fmralign.tests.utils import sample_subjects


def test_alignment_template():
    """Test template alignment."""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    algo = GroupAlignment(labels=labels)
    algo.fit(X, y="template")

    assert len(algo.fitted_estimators) == len(X)
    assert algo.template.shape == X[0].shape
    for i, x in enumerate(X):
        transformed = algo._transform_one_array(x, algo.fitted_estimators[i])
        assert_array_equal(transformed, x)


def test_alignment_target():
    """Test alignment to a target"""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    target = X[0]
    algo = GroupAlignment(labels=labels)
    algo.fit(X, y=target)

    assert len(algo.fitted_estimators) == len(X)
    assert algo.template is None
    for i, x in enumerate(X):
        transformed = algo._transform_one_array(x, algo.fitted_estimators[i])
        assert_array_equal(transformed, x)


def test_transform():
    """Test transform method."""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    algo = GroupAlignment(labels=labels)
    algo.fit(X)

    transformed_arrays = algo.transform(X)
    for i, x in X.items():
        assert_array_equal(transformed_arrays[i], x)
