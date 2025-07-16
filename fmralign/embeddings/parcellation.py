import nibabel as nib
import numpy as np
from nilearn.image import smooth_img
from nilearn.surface import SurfaceImage
from nilearn.regions import Parcellations
from nilearn._utils.niimg_conversions import check_same_fov
from nilearn.masking import apply_mask_fmri
from collections import defaultdict
from scipy.sparse import coo_matrix


def get_labels(
    imgs, n_pieces, masker, clustering="ward", smoothing_fwhm=5, verbose=0
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
    clustering: string or 3D Niimg
        If you aim for speed, choose k-means (and check kmeans_smoothing_fwhm parameter)
        If you want spatially connected and/or reproducible regions use 'ward'
        If you want balanced clusters (especially from timeseries) used 'hierarchical_kmeans'
        If 3D Niimg, image used as predefined clustering, n_pieces is ignored
    n_pieces: int
        number of different labels
    masker: a fitted instance of NiftiMasker or MultiNiftiMasker
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
        _, counts = np.unique(labels, return_counts=True)
        print(f"The alignment will be applied on parcels of sizes {counts}")

    return labels


def sparse_clusters_parcellation(labels):
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

    n = len(labels)

    # Create a dictionary mapping each value to its indices
    value_to_indices = defaultdict(list)
    for i, val in enumerate(labels.tolist()):
        value_to_indices[val].append(i)

    # Create lists to store indices and values for the sparse matrix
    rows = []
    cols = []

    # For each value, add all pairs of indices where that value appears
    for indices in value_to_indices.values():
        for i in indices:
            rows += [i] * len(indices)
            cols += indices

    # Convert to numpy arrays
    rows = np.array(rows)
    cols = np.array(cols)
    values = np.ones(len(rows), dtype=bool)

    # Create sparse COO matrix
    sparse_matrix = coo_matrix((values, (rows, cols)), shape=(n, n))

    return sparse_matrix
