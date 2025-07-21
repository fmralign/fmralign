# -*- coding: utf-8 -*-
import datetime
from collections import defaultdict
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.image import new_img_like, smooth_img
from nilearn.masking import apply_mask_fmri, intersect_masks
from nilearn.regions.parcellations import Parcellations
from nilearn.surface import SurfaceImage
from sklearn.exceptions import NotFittedError


def _intersect_clustering_mask(clustering, mask):
    """Take 3D Niimg clustering and bigger mask, output reduced mask."""
    dat = clustering.get_fdata()
    new_ = np.zeros_like(dat)
    new_[dat > 0] = 1
    clustering_mask = new_img_like(clustering, new_)
    return intersect_masks(
        [clustering_mask, mask], threshold=1, connected=True
    )


def _make_parcellation(
    imgs, clustering, n_pieces, masker, smoothing_fwhm=5, verbose=0
):
    """Compute a parcellation of the data.

    Use nilearn Parcellation class in our pipeline. It is used to find local
    regions of the brain in which alignment will be later applied. For
    alignment computational efficiency, regions should be of hundreds of
    voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    clustering: string or 3D Niimg
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        If you want balanced clusters (especially from timeseries) used 'hierarchical_kmeans'
        If 3D Niimg, image used as predefined clustering, n_pieces is ignored
    n_pieces: int
        number of different labels
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    smoothing_fwhm: None or int
        By default 5mm smoothing will be applied before kmeans clustering to have
        more compact clusters (but this will not change the data later).
        To disable this option, this parameter should be None.

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    # check if clustering is provided
    if isinstance(clustering, nib.nifti1.Nifti1Image):
        check_same_fov(masker.mask_img_, clustering)
        labels = apply_mask_fmri(clustering, masker.mask_img_).astype(int)

    elif isinstance(clustering, SurfaceImage):
        labels = masker.transform(clustering).astype(int)

    # otherwise check it's needed, if not return 1 everywhere
    elif n_pieces == 1:
        labels = np.ones(
            int(masker.mask_img_.get_fdata().sum()), dtype=np.int8
        )

    # otherwise check requested clustering method
    elif isinstance(clustering, str) and n_pieces > 1:
        if (clustering in ["kmeans", "hierarchical_kmeans"]) and (
            smoothing_fwhm is not None
        ):
            images_to_parcel = smooth_img(imgs, smoothing_fwhm)
        else:
            images_to_parcel = imgs
        parcellation = Parcellations(
            method=clustering,
            n_parcels=n_pieces,
            mask=masker,
            scaling=False,
            n_iter=20,
            verbose=verbose,
        )
        try:
            parcellation.fit(images_to_parcel)
        except ValueError as err:
            errmsg = (
                f"Clustering method {clustering} should be supported by "
                "nilearn.regions.Parcellation or a 3D Niimg."
            )
            err.args += (errmsg,)
            raise err
        labels = masker.transform(parcellation.labels_img_).astype(int)

    if verbose > 0:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    return labels


def _sparse_cluster_matrix(arr):
    """
    Creates a sparse matrix where element (i,j) is 1 if arr[i] == arr[j], 0 otherwise.

    Parameters
    ----------
    arr: torch.Tensor of shape (n,)
        1D array of integers

    Returns
    -------
    sparse_matrix: sparse torch.Tensor of shape (len(arr), len(arr))
    """

    n = len(arr)

    # Create a dictionary mapping each value to its indices
    value_to_indices = defaultdict(list)
    for i, val in enumerate(arr.tolist()):
        value_to_indices[val].append(i)

    # Create lists to store indices and values for the sparse matrix
    rows = []
    cols = []

    # For each value, add all pairs of indices where that value appears
    for indices in value_to_indices.values():
        for i in indices:
            rows += [i] * len(indices)
            cols += indices

    # Convert to tensors
    rows = torch.tensor(rows)
    cols = torch.tensor(cols)
    values = torch.ones(len(rows), dtype=torch.bool)

    # Create sparse tensor
    sparse_matrix = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]),
        values=values,
        size=(n, n),
    ).coalesce()

    return sparse_matrix


def save_alignment(alignment_estimator, output_path):
    """Save the alignment estimator object to a file.

    Parameters
    ----------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The alignment estimator object to be saved.
        It should be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`.
        The object should have been fitted before saving.
    output_path : str or Path
        Path to the file or directory where the model will be saved.
        If a directory is provided, the model will be saved with a
        timestamped filename in that directory.
        If a file is provided, the model will be saved with that filename.

    Raises
    ------
    NotFittedError
        If the alignment estimator has not been fitted yet.
    ValueError
        If the output path is not a valid file or directory.
    """
    if not hasattr(alignment_estimator, "fitted_estimators"):
        raise NotFittedError(
            "This instance has not been fitted yet. "
            "Please call 'fit' before 'save'."
        )

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"alignment_estimator_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        joblib.dump(alignment_estimator, output_path / suffix)

    else:
        joblib.dump(alignment_estimator, output_path)


def load_alignment(input_path):
    """Load an alignment estimator object from a file.

    Parameters
    ----------
    input_path : str or Path
        Path to the file or directory from which the model will be loaded.
        If a directory is provided, the latest .pkl file in that directory
        will be loaded.

    Returns
    -------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The loaded alignment estimator object.
        It will be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`, depending on what was saved.

    Raises
    ------
    ValueError
        If no .pkl files are found in the directory or if the input path is not
        a valid file or directory.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # If it's a directory, look for the latest .pkl file
        pkl_files = list(input_path.glob("*.pkl"))
        if not pkl_files:
            raise ValueError(
                f"No .pkl files found in the directory: {input_path}"
            )
        input_path = max(pkl_files, key=lambda x: x.stat().st_mtime)

    return joblib.load(input_path)
