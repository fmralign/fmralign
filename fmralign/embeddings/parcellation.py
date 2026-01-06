import copy
import warnings

import nibabel as nib
import numpy as np
from nilearn.image import smooth_img
from nilearn.image.image import check_same_fov
from nilearn.maskers import (
    MultiNiftiMasker,
    MultiSurfaceMasker,
    NiftiMasker,
    SurfaceMasker,
)
from nilearn.masking import apply_mask_fmri
from nilearn.regions import Parcellations
from nilearn.surface import SurfaceImage
from scipy.sparse import csc_matrix


def _convert_to_multi_masker(masker):
    """Convert a NiftiMasker to a MultiNiftiMasker or a SurfaceMasker to a
    MultiSurfaceMasker.

    Note: A deep copy of the masker is created to avoid modifying the original
    masker. The n_jobs attribute is set to 1 to ensure compatibility with
    multi-maskers.

    Parameters
    ----------
    masker: NiftiMasker or SurfaceMasker
        The masker to convert.

    Returns
    -------
    multi_masker: MultiNiftiMasker or MultiSurfaceMasker
        The converted masker.
    """

    multi_masker = copy.deepcopy(masker)
    if isinstance(masker, NiftiMasker):
        multi_masker.__class__ = MultiNiftiMasker
        multi_masker.n_jobs = 1
        return multi_masker
    elif isinstance(masker, SurfaceMasker):
        multi_masker.__class__ = MultiSurfaceMasker
        multi_masker.n_jobs = 1
        return multi_masker
    else:
        raise ValueError("Masker must be a NiftiMasker or SurfaceMasker.")


def get_labels(
    imgs, masker, n_pieces=1, clustering="ward", smoothing_fwhm=5, verbose=0
):
    """Generate an array of labels for each voxel in the data.

    Use nilearn Parcellation class in our pipeline. It is used to find local
    regions of the brain in which alignment will be later applied. For
    alignment computational efficiency, regions should be of hundreds of
    voxels.

    Parameters
    ----------
    imgs: Niimgs
        data to cluster
    n_pieces: int
        number of different labels
    masker: a fitted instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    clustering: string or 3D Niimg
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        If you want balanced clusters (especially from timeseries) used 'hierarchical_kmeans'
        If 3D Niimg, image used as predefined clustering, n_pieces is ignored
    smoothing_fwhm: None or int
        By default 5mm smoothing will be applied before kmeans clustering to have
        more compact clusters (but this will not change the data later).
        To disable this option, this parameter should be None.
    verbose: int, default=0
        Verbosity level.

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    # check if clustering is provided
    if isinstance(clustering, nib.nifti1.Nifti1Image):
        if n_pieces != 1:
            warnings.warn("Clustering image provided, n_pieces ignored.")
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

        if isinstance(masker, (NiftiMasker, SurfaceMasker)):
            warnings.warn(
                (
                    "Converting masker to multi-masker for compatibility"
                    " with Nilearn Parcellations class. This conversion does"
                    " not affect the original masker. "
                    "See https://github.com/nilearn/nilearn/issues/5926"
                    " for more details."
                )
            )
            masker_ = _convert_to_multi_masker(masker)
        else:
            masker_ = masker
        parcellation = Parcellations(
            method=clustering,
            n_parcels=n_pieces,
            mask=masker_,
            scaling=False,
            n_iter=20,
            verbose=verbose,
        )
        parcellation.fit(images_to_parcel)
        labels = masker.transform(parcellation.labels_img_).astype(int)

    if verbose > 0:
        _, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    return labels


def get_adjacency_from_labels(labels):
    """
    Creates a sparse matrix where element (i,j) is 1
    if labels[i] == labels[j], 0 otherwise.

    Parameters
    ----------
    labels: ndarray of shape (n,)
        1D array of integers

    Returns
    -------
    sparse_matrix: sparse scipy.sparse.coo_matrix
        of shape (len(labels), len(labels))
    """

    n_rows = len(labels)
    unique_labels, col = np.unique(labels, return_inverse=True)
    n_cols = len(unique_labels)
    data = np.ones(n_rows, dtype=bool)
    row = np.arange(n_rows)
    incidence_matrix = csc_matrix(
        (data, (row, col)),
        shape=(n_rows, n_cols),
    )

    return (incidence_matrix @ incidence_matrix.T).tocoo()
