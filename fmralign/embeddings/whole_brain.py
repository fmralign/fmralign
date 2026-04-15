import numpy as np
import robust_laplacian as rl
import scipy.sparse.linalg as sla
from nibabel import Nifti1Image
from nilearn.image import resampling
from nilearn.masking import load_mask_img
from nilearn.surface import InMemoryMesh
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

    Given a mask image (Nifti1Image) or a surface mesh (InMemoryMesh),
    this function computes the Laplace-Beltrami eigenmodes on the vertices
    defined by the mask or mesh. It solves a generalized eigenvalue problem to return
    the first `k` eigenmodes.

    Note: The zero-order eigenmode is skipped.

    Parameters
    ----------
    mask_img : Nifti1Image or InMemoryMesh
        Input mask image or surface mesh.
    k : int
        Number of eigenmodes to compute.
    sigma : float, optional
        Stability parameter passed to the sparse solver, by default 1e-8.

    Returns
    -------
    evecs : ndarray of shape (k, n_vertices)
        The computed Laplacian eigenmodes.
    """
    if isinstance(mask_img, Nifti1Image):
        mask_data, _ = load_mask_img(mask_img)
        V = np.array(np.nonzero(mask_data)).T
        L, M = rl.point_cloud_laplacian(V)
    elif isinstance(mask_img, InMemoryMesh):
        V = np.asarray(mask_img.coordinates, dtype="<f4")
        F = np.asarray(mask_img.faces, dtype="<f4")
        L, M = rl.mesh_laplacian(V, F)
    else:
        raise ValueError("mask_img must be a Nifti1Image or an InMemoryMesh.")

    _, evecs = sla.eigsh(L, k + 1, M, sigma=sigma)

    return evecs.T[1:]  # skip first eigenvector
