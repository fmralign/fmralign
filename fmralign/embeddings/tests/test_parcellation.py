import nibabel as nib
import numpy as np
import pytest
from nilearn.maskers import NiftiMasker, SurfaceMasker
from numpy.testing import assert_array_equal

from fmralign.embeddings.parcellation import (
    get_adjacency_from_labels,
    get_labels,
)
from fmralign.tests.utils import random_niimg, surf_img


def test_get_labels():
    """Test get_labels function on Nifti, and surfs
    with various clusterings."""
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans", "rena"]

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (
            get_labels(img, masker, 1, clustering_method)
            == masker.transform(mask_img)
        ).all()

        # check n_pieces = 2 find right clustering
        labels = get_labels(img, masker, 2, clustering_method)
        assert len(np.unique(labels)) == 2

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert get_labels(img, masker, 0, clustering_method)

    clustering = nib.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4)
    )

    # check that 3D Niimg clusterings override n_pieces
    for n_pieces in [0, 1, 2]:
        labels = get_labels(img, masker, n_pieces, clustering)
        assert len(np.unique(labels)) == 2

    # check surface image
    img = surf_img(5)
    masker = SurfaceMasker().fit(img)
    labels = get_labels(img, masker, n_pieces)
    assert len(np.unique(labels)) == 2


def test_get_adjacency_from_labels():
    """Test get_adjacency_from_labels on 2 clusters."""
    labels = np.array([1, 1, 2, 2, 2])
    sparse_matrix = get_adjacency_from_labels(labels)

    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )

    assert sparse_matrix.shape == (5, 5)
    assert sparse_matrix.dtype == bool
    assert_array_equal(sparse_matrix.todense(), expected)
