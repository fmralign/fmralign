import pytest
import numpy as np
import nibabel as nib
from fmralign.embeddings.parcellation import get_labels
from fmralign.tests.utils import random_niimg, surf_img
from nilearn.maskers import NiftiMasker, SurfaceMasker


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
