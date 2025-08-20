import numpy as np
from numpy.testing import assert_array_equal

from fmralign.embeddings.whole_brain import get_adjacency_from_mask
from fmralign.tests.utils import random_niimg


def test_get_adjacency_from_mask():
    """Test get_adjacency_from_mask on a simple 2x2 mask."""
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
