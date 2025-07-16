import numpy as np
from sklearn import neighbors
from nilearn.masking import load_mask_img
from nilearn.image import resampling


def get_adjacency_from_mask(mask_img, radius):
    """
    Creates a sparse adjacency matrix from a mask image where each voxel
    is connected to its neighbors within a specified radius.

    Parameters
    ----------
    mask_img: 3D Nifti1Image
        Mask image to define the voxels.
    radius: float
        Radius in mm to define the neighborhood for each voxel.


    Returns
    -------
    sparse_matrix: sparse torch.Tensor of shape (n_voxels, n_voxels)
    """
    mask_data, mask_affine = load_mask_img(mask_img)
    mask_coords = np.where(mask_data != 0)
    mask_coords = resampling.coord_transform(
        mask_coords[0],
        mask_coords[1],
        mask_coords[2],
        mask_affine,
    )
    mask_coords = np.asarray(mask_coords).T
    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(mask_coords)

    return A.tocoo().astype(bool)
