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
    """
    Computes the Laplacian embedding of a given mask image or surface mesh.

    Given a mask image (Nifti1Image) or a surface mesh (PolyMesh or InMemoryMesh),
    this function computes the Laplace-Beltrami eigenfunctions on the vertices
    defined by the mask or mesh. It solves a generalized eigenvalue problem to return
    the first `k` eigenfunctions.

    Parameters
    ----------
    mask_img : Nifti1Image or PolyMesh or InMemoryMesh
        Input mask image or surface mesh.
    k : int
        Number of eigenfunctions to compute.
    sigma : float, optional
        Stability parameter passed to the sparse solver, by default 1e-8.

    Returns
    -------
    evecs : ndarray of shape (k, n_vertices)
        The computed Laplacian eigenfunctions.
    """
    if isinstance(mask_img, Nifti1Image):
        mask_data, _ = load_mask_img(mask_img)
        V = np.array(np.nonzero(mask_data)).T
        L, M = rl.point_cloud_laplacian(V)
    elif isinstance(mask_img, PolyMesh):
        V = np.vstack(
            [
                np.asarray(part.coordinates, dtype="<f4")
                for part in mask_img.parts.values()
            ]
        )
        F = np.vstack(
            [
                np.asarray(part.faces, dtype="<f4")
                for part in mask_img.parts.values()
            ]
        )
        L, M = L, M = rl.mesh_laplacian(V, F)
    else:
        V = np.asarray(mask_img.coordinates, dtype="<f4")
        F = np.asarray(mask_img.faces, dtype="<f4")
        L, M = rl.mesh_laplacian(V, F)

    _, evecs = sla.eigsh(L, k, M, sigma=sigma)
    return evecs.T
