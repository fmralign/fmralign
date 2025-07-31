from numpy.testing import assert_array_equal

from fmralign import PairwiseAlignment
from fmralign.tests.utils import sample_subjects
from fmralign.methods import Identity


def test_fit():
    """Test pairwise alignment fitting."""
    subjects_data, labels = sample_subjects(n_subjects=2)

    algo = PairwiseAlignment(labels=labels)
    algo.fit(subjects_data[0], subjects_data[1])

    assert_array_equal(algo.labels_, labels)
    assert isinstance(algo.method_, Identity)
    assert algo.fitted_estimator is not None


def test_transform():
    """Test transform method."""
    subjects_data, labels = sample_subjects(n_subjects=2)

    algo = PairwiseAlignment(labels=labels)
    algo.fit(subjects_data[0], subjects_data[1])

    transformed_array = algo.transform(subjects_data[0])
    assert_array_equal(transformed_array, subjects_data[0])
