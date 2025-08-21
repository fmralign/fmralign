# -*- coding: utf-8 -*-
import pytest
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError

from fmralign import GroupAlignment
from fmralign._utils import (
    load_alignment,
    save_alignment,
)
from fmralign.tests.utils import sample_subjects


def test_saving_and_loading(tmp_path):
    """Test saving and loading utilities."""
    subjects_data, labels = sample_subjects()
    X = dict(enumerate(subjects_data))

    algo = GroupAlignment(labels=labels)

    # Check that there is an error when trying to save without fitting
    with pytest.raises(NotFittedError):
        save_alignment(algo, tmp_path)

    # Fit the model
    algo.fit(X)
    # Save the model
    save_alignment(algo, tmp_path)
    # Load the model
    loaded_model = load_alignment(tmp_path)

    # Check that the transformed arrays are the same
    transformed = loaded_model.transform(X)
    assert_array_equal(transformed[0], subjects_data[0])
