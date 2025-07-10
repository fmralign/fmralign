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

sub_ids = ["sub-01", "sub-02", "sub-04"]
files, df, mask = fetch_ibc_subjects_contrasts(sub_ids)

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
# in the final data, we will take  advantage of these pseudo-duplicates later.

# Let's organize our data into training and testing sets:
# * Training set: Files from the first two subjects (sub-01, sub-02) with both
#   AP and PA acquisitions. These will serve as our source subjects for
#   alignment.
# * Testing set: Files from the third subject (sub-04) with both AP and PA
#   acquisitions, which will be our target subject for evaluating alignment
#   quality.

# Split data into source and target subjects
source_train = [concat_imgs(files[i]) for i in range(2)]  # sub-01 and sub-02
target = concat_imgs(files[2])  # sub-04

###############################################################################
# Choose the number of regions for local alignment
# ------------------------------------------------
# First, as we will proceed to local alignment we choose a suitable number of
# regions so that each of them is approximately 100 voxels wide. Then our
# estimator will first make a functional clustering of voxels based on train
# data to divide them into meaningful regions.

import numpy as np

n_voxels = roi_masker.mask_img_.get_fdata().sum()
print(f"The chosen region of interest contains {n_voxels} voxels")
n_pieces = int(np.round(n_voxels / 100))
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

import matplotlib.pyplot as plt

from fmralign.metrics import score_voxelwise
from fmralign.template_alignment import TemplateAlignment

fig, axes = plt.subplots(3, 1, figsize=(8, 10))

methods = ["scaled_orthogonal", "optimal_transport"]

for i, method in enumerate(methods):
    alignment_estimator = TemplateAlignment(
        alignment_method=method, n_pieces=n_pieces, masker=roi_masker
    )
    alignment_estimator.fit(source_train)
    target_pred = alignment_estimator.transform(target)

    # derive correlation between prediction, test
    method_error = score_voxelwise(
        target, target_pred, masker=roi_masker, loss="corr"
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
        axes=axes[i],
        colorbar=False,
    )

###############################################################################
# Next, let's compare the template-based alignment methods to the Shared
# Response Model (SRM) method. SRM computes a shared response space from
# different subjects to a particular task, and then projects individual subject
# data into this shared space.

###############################################################################
# Initialize the IdentifiableFastSRM model. This version of SRM ensures that
# the solution is unique.
#

from fastsrm.identifiable_srm import IdentifiableFastSRM

srm = IdentifiableFastSRM(
    n_components=30,
    n_iter=10,
)

# #############################################################################
# For the SRM method, we will use the same training data as before
# (source subjects: sub-01 and sub-02) and the target subject (sub-04), and we
# will divide our source and target subjects into training and testing sets,
# leveraging the AP and PA acquisitions.

# The training set:
# * source train: AP acquisitions from source subjects (sub-01, sub-02).
#   These will be projected into a shared response space.
# * target train: AP acquisitions from the target subject (sub-04).
#   These define the target shared space.
#

source_subjects = [sub for sub in sub_ids if sub != "sub-04"]
source_train = [
    concat_imgs(df[(df.subject == sub) & (df.acquisition == "ap")].path.values)
    for sub in source_subjects
]
target_train = concat_imgs(
    df[(df.subject == "sub-04") & (df.acquisition == "ap")].path.values
)

# The testing set:
# * source test: acquisitions from source subjects (sub-01, sub-02).
#   These will be projected into the shared space and transformed to predict
#   the target.
# * target test: PA acquisitions from the target subject (sub-04).
#   These serve as ground truth to evaluate prediction accuracy.
#

source_test = [
    concat_imgs(df[(df.subject == sub) & (df.acquisition == "pa")].path.values)
    for sub in source_subjects
]
target_test = concat_imgs(
    df[(df.subject == "sub-04") & (df.acquisition == "pa")].path.values
)

################################################################################
# Fit the SRM model
# -----------------

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
    axes=axes[len(methods)],
    colorbar=True,
)

plt.tight_layout()
plt.show()

###############################################################################
# Summary:
# --------
# We compared TemplateAlignment methods (scaled orthogonal, optimal transport)
# with SRM-based alignment on visual cortex activity.
