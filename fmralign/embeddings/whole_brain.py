import numpy as np
import robust_laplacian as rl
import scipy.sparse.linalg as sla
from nibabel import Nifti1Image
from nilearn.image import resampling
from nilearn.masking import load_mask_img
from nilearn.surface import PolyMesh
from sklearn import neighbors


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


def get_laplacian_embedding(mask_img, k, sigma=1e-8):
    if isinstance(mask_img, Nifti1Image):
        mask_data, _ = load_mask_img(mask_img)
        V = np.array(np.nonzero(mask_data)).T
        L, M = rl.point_cloud_laplacian(V)
    else:
        mesh = (
            mask_img
            if isinstance(mask_img, PolyMesh)
            else PolyMesh(**mask_img)
        )
        V = np.vstack(
            [
                np.asarray(part.coordinates, dtype="<f4")
                for part in mesh.parts.values()
            ]
        )
        F = np.vstack(
            [
                np.asarray(part.faces, dtype="<f4")
                for part in mesh.parts.values()
            ]
        )
        L, M = L, M = rl.mesh_laplacian(V, F)

    _, evecs = sla.eigsh(L, k, M, sigma=sigma)
    return evecs.T
