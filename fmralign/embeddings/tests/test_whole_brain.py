import numpy as np
from nilearn.conftest import _make_mesh
from numpy.testing import assert_array_equal

from fmralign.embeddings.whole_brain import (
    get_adjacency_from_mask,
    get_laplacian_embedding,
)
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


def test_get_laplacian_embedding():
    """Test get_laplacian_embedding on simple configurations."""

    # Volume
    _, mask_img = random_niimg((5, 4, 3))
    out = get_laplacian_embedding(mask_img, k=3)
    assert out.shape == (3, 60)
    assert out.dtype == np.float64

    # Surface
    mesh = _make_mesh()

    # PolyMesh
    out = get_laplacian_embedding(mesh, k=3)
    assert out.shape == (3, mesh.n_vertices)
    assert out.dtype == np.float64

    # Single mesh
    left_mesh = mesh.parts["left"]
    out = get_laplacian_embedding(left_mesh, k=3)
    assert out.shape == (3, left_mesh.n_vertices)
    assert out.dtype == np.float64
