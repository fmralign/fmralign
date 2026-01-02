"""
Whole Brain Alignment without Parcellation
==========================================

In this example, we show how parcellation can lead to undesirable boundary
artifacts when computing alignments. We then introduce soft constraints based on
the geometry of the cortical surface to alleviate these issues and showcase how
the :class:`~fmralign.methods.optimal_transport.OptimalTransport` can be used
in conjunction with the `geomloss`_ backend (see :footcite:t:`Feydy2019`) to
perform parcellation-free whole-brain alignment.


.. _geomloss: https://www.kernel-operations.io/geomloss/
"""

###############################################################################
# Loading and projecting the data
# ---------------------------------
# We load fMRI images from two subjects from the IBC dataset and project them
# on the cortical surface using nilearn. The details of this step are more
# thoroughly described in the :doc:`surface alignment example <plot_surf_alignment>`.

from nilearn.datasets import load_fsaverage
from nilearn.image import concat_imgs
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

files, df, _ = fetch_ibc_subjects_contrasts(["sub-01", "sub-04"])
fsaverage_meshes = load_fsaverage("fsaverage4")


def project_to_surface(img):
    """Util function for loading and projecting volumetric images."""
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img,
    )
    return surface_image


source_train = concat_imgs(
    df[(df.subject == "sub-01") & (df.acquisition == "ap")].path.values
)
target_train = concat_imgs(
    df[(df.subject == "sub-04") & (df.acquisition == "ap")].path.values
)

surf_source_train = project_to_surface(source_train)
surf_target_train = project_to_surface(target_train)

masker = SurfaceMasker().fit([surf_source_train, surf_target_train])


###############################################################################
# Computing and plotting a parcellation
# -------------------------------------
# We compute a parcellation for local alignments using the
# :func:`~fmralign.embeddings.parcellation.get_labels`
# function and plot it on the surface using nilearn.

from nilearn import plotting

from fmralign.embeddings.parcellation import get_labels

labels = get_labels(
    [surf_source_train, surf_target_train],
    n_pieces=100,
    masker=masker,
    clustering="ward",
)

clustering_img = masker.inverse_transform(labels)

plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"],
    roi_map=clustering_img,
    hemi="left",
    view="lateral",
    title="Ward parcellation on the left hemisphere",
)

###############################################################################
# Fitting the alignment operator
# ------------------------------
# We use the computed labels to fit a local alignment procedure using
# Optimal Transport (OT).

from fmralign import PairwiseAlignment

data_source_train = masker.transform(surf_source_train)
data_target_train = masker.transform(surf_target_train)

alignment_estimator = PairwiseAlignment(
    method="ot",
    labels=labels,
)

alignment_estimator.fit(data_source_train, data_target_train)

#################################################################################
# Simulating an activation patch
# ------------------------------
# We simulate some data on the source subject by generating a spherical
# patch of activation on the temporal lobe.

import numpy as np

center_vertex_idx = 2150

simulated = np.zeros((1, data_source_train.shape[1]))
vertices = surf_source_train.mesh.parts["left"].coordinates
distances = np.linalg.norm(vertices - vertices[center_vertex_idx], axis=1)
simulated[0, : simulated.shape[1] // 2] = distances < 15

plotting.plot_surf_roi(
    fsaverage_meshes["inflated"],
    masker.inverse_transform(simulated),
    title="Simulated activation on the source subject",
)

#################################################################################
# Transporting the activation patch
# ---------------------------------
# We use the fitted alignment operator to transport the simulated patch of
# activation to the target subject. We then visualize the voxels that have been
# affected by the transport operation and overlay the contours of the
# ROIs. We observe that the parcellation
# introduces some undesirable boundary effects. In particular, we observe that
# some voxels are completely unaffected on the inferior temporal gyrus.

import matplotlib.pyplot as plt

affected_voxels = masker.inverse_transform(
    alignment_estimator.transform(simulated) != 0
)

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

plotting.plot_surf_roi(
    fsaverage_meshes["inflated"],
    affected_voxels,
    title="Transported Voxels",
    axes=ax,
    colorbar=False,
)
plotting.plot_surf_contours(
    fsaverage_meshes["inflated"],
    masker.inverse_transform(labels),
    axes=ax,
)
#################################################################################
# Soft constraints using the Laplacian embedding
# ----------------------------------------------
# To alleviate the boundary effects introduced by parcellation, we can
# introduce soft constraints based on the geometry of the cortical surface.
# Here, we compute the eigenfunctions of the Laplace-Beltrami operator on the
# cortical mesh using the :func:`~fmralign.embeddings.whole_brain.get_laplacian_embedding`
# function. These eigenfunctions intrinsically capture the geometry of the surface
# and can be used as additional features during alignment to enforce locality
# constraints in a smooth manner. We normalize the embedding and visualize a
# few eigenfunctions on the cortical surface. Notice how the oscillatory patterns
# of the eigenfunctions follow the geometry of the cortical surface and increase
# in frequency as we move to higher index eigenfunctions.

from fmralign.embeddings.whole_brain import get_laplacian_embedding

geom_embedding = get_laplacian_embedding(masker.mask_img_.mesh, k=100)
geom_embedding = geom_embedding / np.max(np.abs(geom_embedding), axis=0)

fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"})

for i, idx in enumerate([2, 20, 100]):
    plotting.plot_surf_stat_map(
        fsaverage_meshes["pial"],
        masker.inverse_transform(geom_embedding[idx - 1]),
        title=f"Eigenvector {idx}",
        axes=ax[i],
        colorbar=False,
    )


#################################################################################
# Visualizing the cost associated with the geometric embedding
# ------------------------------------------------------------
# To better understand how the geometric embedding can help enforce locality
# constraints during alignment, we visualize the squared cost associated
# with the embedding. We select the vertex at the center of the simulated patch
# and compute the squared cost to all other vertices on the cortical surface.
# We observe that the cost increases smoothly as we move away from the center
# of the patch, demonstrating how eigenfunctions functionally encode the
# geometry of the cortical surface.

squared_cost = (
    (geom_embedding - geom_embedding[:, center_vertex_idx][:, None]) ** 2
).sum(axis=0)

plotting.plot_surf_stat_map(
    fsaverage_meshes["pial"],
    masker.inverse_transform(squared_cost),
    title="Squared cost to the center of the patch",
)

#################################################################################
# Fitting the alignment with soft constraints
# -------------------------------------------
# We can now fit the alignment operator at the whole-brain level by
# concatenating the geometric embedding with the functional data.
# The :class:`~fmralign.methods.optimal_transport.OptimalTransport` method
# will automatically use the geomloss backend to batch computations whenever
# the number of voxels is over 1000. This backend uses a scaling parameter
# to progressively reduce the entropic regularization during optimization.
# In this example, we set the scaling to 0.5 to speed up convergence.
# We concatenate the geometric embedding with the functional data
# using a weighting parameter alpha to balance the relative influence of the data
# versus the geometry during alignment.
# We set alpha to 0.1, indicating a stronger weighting for the geometry compared to the activation data.

from fmralign.methods.optimal_transport import OptimalTransport

alignment_estimator = PairwiseAlignment(method=OptimalTransport(scaling=0.5))

alpha = 0.1

X = np.vstack(
    [alpha * masker.transform(surf_source_train), (1 - alpha) * geom_embedding]
)
Y = np.vstack(
    [alpha * masker.transform(surf_target_train), (1 - alpha) * geom_embedding]
)
alignment_estimator.fit(X, Y)
affected_voxels = masker.inverse_transform(
    alignment_estimator.transform(simulated) != 0
)


#################################################################################
# Visualizing the transported activation with soft constraints
# ------------------------------------------------------------
# We use the fitted alignment operator to transport the simulated patch of
# activation to the target subject. We then visualize the voxels that have been
# affected by the transport operation and overlay the contours of the
# original activation patch. Compared to the previous result, we see that using
# soft constraints based on the Laplace-Beltrami operators alleviates the
# boundary effects introduced by parcellation. In particular, we see that the
# transported activation avoids any abrupt cut-offs.

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})

plotting.plot_surf_roi(
    fsaverage_meshes["inflated"],
    affected_voxels,
    title="Transported voxels with soft constraints",
    axes=ax,
)
plotting.plot_surf_contours(
    fsaverage_meshes["inflated"],
    masker.inverse_transform(simulated != 0),
    title="Transported voxels with soft constraints",
    axes=ax,
)

###############################################################################
# Hyperparameter tuning via cross-validation
# ------------------------------------------
# We can use cross-validation to properly select the optimal weighting parameter
# alpha between the functional data and the geometric embedding. We rely on
# scikit-learn's :class:`GridSearchCV` to search over a grid of alpha values
# and select the one that maximizes the R² score on held-out data. In order
# to ease the process, the geometric embedding is directly passed to the
# :class:`OptimalTransport` at initialization.

from sklearn.model_selection import GridSearchCV

estimator = PairwiseAlignment(
    method=OptimalTransport(scaling=0.5, evecs=geom_embedding)
)

grid = GridSearchCV(
    estimator=estimator,
    param_grid={"method__alpha": [0.1, 0.5, 0.9]},
    scoring="r2",
    cv=5,
)
grid.fit(data_source_train, data_target_train)

# %%
# Our initial choice of alpha was indeed optimal
print(f"Best parameter : {grid.best_params_}")
