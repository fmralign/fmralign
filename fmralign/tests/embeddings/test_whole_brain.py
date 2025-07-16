from fmralign.embeddings.whole_brain import get_adjacency_from_mask
from fmralign.tests.utils import random_niimg
from numpy.testing import assert_array_equal
import numpy as np


def test_get_adjacency_from_mask():
    _, mask_img = random_niimg((2, 2, 1))
    out = get_adjacency_from_mask(mask_img, radius=1)
    expected = np.matrix(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert out.shape == (4, 4)
    assert out.dtype == bool
    assert_array_equal(out.todense(), expected)
