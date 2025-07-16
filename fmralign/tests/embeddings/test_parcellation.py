import pytest
import numpy as np
import nibabel as nib
from fmralign.embeddings.parcellation import (
    get_labels,
    get_adjacency_from_labels,
)
from fmralign.tests.utils import random_niimg, surf_img
from nilearn.maskers import NiftiMasker, SurfaceMasker
from numpy.testing import assert_array_equal


def test_get_labels():
    """Test get_labels function on Nifti, and surfs
    with various clusterings."""
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans", "rena"]

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (
            get_labels(img, 1, masker, clustering_method)
            == masker.transform(mask_img)
        ).all()

        # check n_pieces = 2 find right clustering
        labels = get_labels(img, 2, masker, clustering_method)
        assert len(np.unique(labels)) == 2

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert get_labels(img, 0, masker, clustering_method)

    clustering = nib.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4)
    )

    # check that 3D Niimg clusterings override n_pieces
    for n_pieces in [0, 1, 2]:
        labels = get_labels(img, n_pieces, masker, clustering)
        assert len(np.unique(labels)) == 2

    # check surface image
    img = surf_img(5)
    masker = SurfaceMasker().fit(img)
    labels = get_labels(img, 2, masker)
    assert len(np.unique(labels)) == 2


def test_get_adjacency_from_labels():
    """Test _sparse_cluster_matrix on 2 clusters."""
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
