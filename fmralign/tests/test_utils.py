# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
import pytest
import torch
from nilearn.maskers import NiftiMasker
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError

from fmralign import GroupAlignment
from fmralign._utils import (
    _make_parcellation,
    _sparse_cluster_matrix,
    load_alignment,
    save_alignment,
)
from fmralign.tests.utils import random_niimg, sample_subjects


def test_make_parcellation():
    # make_parcellation is built on Nilearn which already
    # has several test for its Parcellation class
    # here we test just the call of the API is right on a simple example
    img, mask_img = random_niimg((7, 6, 8, 5))
    masker = NiftiMasker(mask_img=mask_img).fit()

    methods = ["kmeans", "ward", "hierarchical_kmeans", "rena"]

    for clustering_method in methods:
        # check n_pieces = 1 gives out ones of right shape
        assert (
            _make_parcellation(img, clustering_method, 1, masker)
            == masker.transform(mask_img)
        ).all()

        # check n_pieces = 2 find right clustering
        labels = _make_parcellation(img, clustering_method, 2, masker)
        assert len(np.unique(labels)) == 2

        # check that not inputing n_pieces yields problems
        with pytest.raises(Exception):
            assert _make_parcellation(img, clustering_method, 0, masker)

    clustering = nib.Nifti1Image(
        np.hstack([np.ones((7, 3, 8)), 2 * np.ones((7, 3, 8))]), np.eye(4)
    )

    # check 3D Niimg clusterings
    for n_pieces in [0, 1, 2]:
        labels = _make_parcellation(img, clustering, n_pieces, masker)
        assert len(np.unique(labels)) == 2

    # check warning if a parcel is too big
    with pytest.warns(UserWarning):
        clustering = nib.Nifti1Image(
            np.hstack([np.ones(2000), 4 * np.ones(800)]), np.eye(4)
        )
        _make_parcellation(img, clustering_method, n_pieces, masker)


def test_sparse_cluster_matrix():
    """Test _sparse_cluster_matrix on 2 clusters."""
    labels = torch.tensor([1, 1, 2, 2, 2])
    sparse_matrix = _sparse_cluster_matrix(labels)

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    assert sparse_matrix.shape == (5, 5)
    assert sparse_matrix.dtype == torch.bool
    assert torch.allclose(sparse_matrix.to_dense(), expected)


def test_saving_and_loading(tmp_path):
    """Test saving and loading utilities."""
    X, labels = sample_subjects()

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
    [transformed] = loaded_model.transform(X, [0])
    assert_array_almost_equal(transformed, X[0])
