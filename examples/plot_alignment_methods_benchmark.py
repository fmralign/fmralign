# -*- coding: utf-8 -*-

"""
Alignment methods benchmark (template-based ROI case)
=====================================================

In this tutorial, we compare various methods of alignment on a pairwise alignment
problem for Individual Brain Charting subjects. For each subject, we have a lot
of functional informations in the form of several task-based
contrast per subject. We will just work here on a ROI.

We mostly rely on python common packages and on nilearn to handle functional
data in a clean fashion.

To run this example, you must launch IPython via ``ipython --matplotlib`` in
a terminal, or use ``jupyter-notebook``.
"""

###############################################################################
#  Retrieve the data
# ------------------
# In this example we use the IBC dataset, which include a large number of
# different contrasts maps for 12 subjects.
# We download the images for subjects sub-01 and sub-02.
# Files is the list of paths for each subjects.
# df is a dataframe with metadata about each of them.

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts

files, df, mask = fetch_ibc_subjects_contrasts(["sub-01", "sub-02"])


###############################################################################
# Extract a mask for the visual cortex from Yeo Atlas
# ---------------------------------------------------
# First, we fetch and plot the complete atlas

from nilearn import datasets, plotting
from nilearn.image import concat_imgs, load_img, new_img_like, resample_to_img

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas = load_img(atlas_yeo_2011.thick_7)

# Select visual cortex, create a mask and resample it to the right resolution

mask_visual = new_img_like(atlas, atlas.get_fdata() == 1)
resampled_mask_visual = resample_to_img(
    mask_visual, mask, interpolation="nearest"
)

# Plot the mask we will use
plotting.plot_roi(
    resampled_mask_visual,
    title="Visual regions mask extracted from atlas",
    cut_coords=(8, -80, 9),
    colorbar=False,
    cmap="Paired",
)

###############################################################################
# Define a masker
# ---------------
# We define a nilearn masker that will be used to handle relevant data.
# For more information, visit :
# 'http://nilearn.github.io/manipulating_images/masker_objects.html'

from nilearn.maskers import NiftiMasker

roi_masker = NiftiMasker(mask_img=resampled_mask_visual).fit()

###############################################################################
# Prepare the data
# ----------------
# For each subject, for each task and conditions, our dataset contains two
# independent acquisitions, similar except for one acquisition parameter, the
# encoding phase used that was either Antero-Posterior (AP) or
# Postero-Anterior (PA). Although this induces small differences
# in the final data, we will take  advantage of these pseudo-duplicates to
# create a training and a testing set that contains roughly the same signals
# but acquired independently.

# The training set, used to learn alignment from source subject toward target:
# * source train: AP contrasts for subject sub-01
# * target train: AP contrasts for subject sub-02
#

source_train = concat_imgs(
    df[df.subject == "sub-01"][df.acquisition == "ap"].path.values
)
target_train = concat_imgs(
    df[df.subject == "sub-02"][df.acquisition == "ap"].path.values
)

# The testing set:
# * source test: PA contrasts for subject one, used to predict
#   the corresponding contrasts of subject sub-01
# * target test: PA contrasts for subject sub-02, used as a ground truth
#   to score our predictions
#

source_test = concat_imgs(
    df[df.subject == "sub-01"][df.acquisition == "pa"].path.values
)
target_test = concat_imgs(
    df[df.subject == "sub-02"][df.acquisition == "pa"].path.values
)

###############################################################################
# Choose the number of regions for local alignment
# ------------------------------------------------
# First, as we will proceed to local alignment we choose a suitable number of
# regions so that each of them is approximately 200 voxels wide. Then our
# estimator will first make a functional clustering of voxels based on train
# data to divide them into meaningful regions.

import numpy as np

n_voxels = roi_masker.mask_img_.get_fdata().sum()
print(f"The chosen region of interest contains {n_voxels} voxels")
n_pieces = int(np.round(n_voxels / 200))
print(f"We will cluster them in {n_pieces} regions")

###############################################################################
# Define the estimators, fit them and do a prediction
# ---------------------------------------------------
# On each region, we search for a transformation R that is either :
#   *  orthogonal, i.e. R orthogonal, scaling sc s.t. ||sc RX - Y ||^2 is minimized
#   *  the optimal transport plan, which yields the minimal transport cost
#       while respecting the mass conservation constraints. Calculated with
#       entropic regularization.
# Then for each method we define the estimator, fit it, predict the new image and plot
# its correlation with the real signal.

from fmralign.metrics import score_voxelwise
from fmralign.template_alignment import TemplateAlignment

methods = ["scaled_orthogonal", "optimal_transport"]

for method in methods:
    alignment_estimator = TemplateAlignment(
        alignment_method=method, n_pieces=n_pieces, masker=roi_masker
    )
    alignment_estimator.fit(source_train)
    target_pred = alignment_estimator.transform(source_test)

    # derive correlation between prediction, test
    method_error = score_voxelwise(
        target_test, target_pred, masker=roi_masker, loss="corr"
    )

    # plot correlation for each method
    aligned_score = roi_masker.inverse_transform(method_error)
    title = f"Correlation of prediction after {method} alignment"
    display = plotting.plot_stat_map(
        aligned_score,
        display_mode="z",
        cut_coords=[-15, -5],
        vmax=1,
        title=title,
    )

###############################################################################
# We can observe that among the two methods, the optimal transport method
# yields a better correlation with the target test data.
#
# Compare to SRM method
# ------------------------
# We now compare the results of the template-based alignment methods to the
# Shared Response Model (SRM) method.

# **Note:** SRM needs multiple source subjects to better estimate the shared
# response.

sub_ids = ["sub-01", "sub-02", "sub-04", "sub-05"]
files, df, mask = fetch_ibc_subjects_contrasts(sub_ids)

# We will use subject 04 as the target subject, and the rest of the subjects as
# source subjects. We will use the *AP* acquisition for training and the *PA*
# acquisition for testing.

source_train = []
for sub in sub_ids:
    if sub not in ["sub-04"]:
        source_train.append(
            concat_imgs(
                df[df.subject == sub][df.acquisition == "ap"].path.values
            )
        )
target_train = concat_imgs(
    df[df.subject == "sub-04"][df.acquisition == "ap"].path.values
)

source_test = []
for sub in sub_ids:
    if sub not in ["sub-04"]:
        source_test.append(
            concat_imgs(
                df[df.subject == sub][df.acquisition == "pa"].path.values
            )
        )
target_test = concat_imgs(
    df[df.subject == "sub-04"][df.acquisition == "pa"].path.values
)

###############################################################################
# Choose the number of regions for local alignment
# ------------------------------------------------
# We'll again divide the ROI into parcels of ~200 voxels each. Then our
# estimator will first make a functional clustering of voxels based on train
# data to divide them into meaningful regions.

n_voxels = roi_masker.mask_img_.get_fdata().sum()
print(f"The chosen region of interest contains {n_voxels} voxels")
n_pieces = int(np.round(n_voxels / 200))
print(f"We will cluster them in {n_pieces} regions")

###############################################################################
# Initialize the IdentifiableFastSRM model, This version of SRM ensures that
# the solution is unique.

import numpy as np
from fastsrm.identifiable_srm import IdentifiableFastSRM

from fmralign.srm import PiecewiseModel

srm = IdentifiableFastSRM(
    n_components=30,
    n_iter=10,
)
piecewise_srm = PiecewiseModel(
    srm=srm,
    n_pieces=n_pieces,
    clustering="ward",
    masker=roi_masker,
)

# Step 1: Fit SRM on training data from source subjects
shared_response = srm.fit_transform(
    [roi_masker.transform(s).T for s in source_train]
)

# Step 2: Freeze the SRM model and add target subject data. This projects the
# target subject data into the shared response space.
srm.aggregate = None
srm.add_subjects([roi_masker.transform(target_train).T], shared_response)

# Step 3: Use SRM to transform new test data from the target subject
aligned_test = srm.transform([roi_masker.transform(target_test).T])
aligned_pred = roi_masker.inverse_transform(
    srm.inverse_transform(aligned_test[0])[0].T
)

# Step 4: Evaluate voxelwise correlation between predicted and true test
# signals.
srm_error = score_voxelwise(
    target_test, aligned_pred, masker=roi_masker, loss="corr"
)
srm_score = roi_masker.inverse_transform(srm_error)
title = "Correlation of prediction after SRM alignment"
display = plotting.plot_stat_map(
    srm_score,
    display_mode="z",
    cut_coords=[-15, -5],
    vmax=1,
    title=title,
)
###############################################################################
# Summary:
# --------
# We compared TemplateAlignment methods (scaled orthogonal, optimal transport)
# with SRM-based alignment on visual cortex activity. While TemplateAlignment
# operates pairwise, SRM uses group information to find a shared representational
# space. SRM may generalize better when multiple subjects are available.
