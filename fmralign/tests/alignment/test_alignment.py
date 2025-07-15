from fmralign import GroupAlignment
from fmralign.tests.utils import sample_subjects
from numpy.testing import assert_array_almost_equal


def test_alignment_template():
    """Test template alignment."""
    X, labels = sample_subjects()

    algo = GroupAlignment(target=None, labels=labels)
    algo.fit(X)

    assert len(algo.fit_) == len(X)
    assert algo.template.shape == X[0].shape
    for i, x in enumerate(X):
        transformed = algo._transform_one_array(x, algo.fit_[i])
        assert_array_almost_equal(transformed, x)


def test_alignment_target():
    """Test alignment to a target"""
    X, labels = sample_subjects()

    target = X[0]
    algo = GroupAlignment(target=target, labels=labels)
    algo.fit(X)

    assert len(algo.fit_) == len(X)
    assert algo.template is None
    for i, x in enumerate(X):
        transformed = algo._transform_one_array(x, algo.fit_[i])
        assert_array_almost_equal(transformed, x)


def test_transform():
    """Test transform method."""
    X, labels = sample_subjects()

    algo = GroupAlignment(labels=labels)
    algo.fit(X)

    transformed_arrays = algo.transform(X, range(len(X)))
    for i, x in enumerate(X):
        assert_array_almost_equal(transformed_arrays[i], x)
