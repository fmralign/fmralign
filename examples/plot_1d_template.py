"""
Template alignment on simulated 1D data
=======================================

In this example, we simulate a group of subjects who share the same
functional organization, but whose functional areas are located at slightly
different places in each brain. This mimics the idiosyncratic topographies
observed in real fMRI data, that anatomical registration cannot correct.

Because the data are simulated, we know the noise-free activation map of each
subject: the *hidden truth*. We use it to compare how well different
strategies predict a left-out subject's activation map from the other
subjects: anatomical averaging, spatial smoothing, and template alignment with
:class:`~fmralign.methods.Procrustes` (see :footcite:t:`Haxby2011`) or
:class:`~fmralign.methods.OptimalTransport`
(see :footcite:t:`Bazeille2019`).
"""

# %%
# Simulating a shared functional organization
# -------------------------------------------
# We consider a 1D brain region of 200 voxels arranged on a ring, which
# avoids border effects. The ring is tiled by 20 functional blocks of 10
# voxels. Each block has its own response profile to the 30 experimental
# conditions used for alignment, and our contrast of interest activates one
# block out of five: these four patches are the hidden truth we want to
# recover.

import numpy as np
from scipy.ndimage import gaussian_filter1d

n_voxels, n_conditions, n_blocks, noise_sd = 200, 30, 20, 0.1
rng = np.random.default_rng(0)

block_widths = np.full(n_blocks, n_voxels / n_blocks)
block_profiles = rng.standard_normal((n_conditions, n_blocks))
block_is_active = np.zeros(n_blocks)
block_is_active[2::5] = 1


def block_of_voxel(widths, shift=0.0):
    """Index of the block covering each voxel of the ring."""
    edges = np.cumsum(widths) / np.sum(widths) * n_voxels
    positions = (np.arange(n_voxels) - shift) % n_voxels
    return np.searchsorted(edges, positions, side="right")


hidden_truth = block_is_active[block_of_voxel(block_widths)]

# %%
# Simulating subjects with idiosyncratic topographies
# ---------------------------------------------------
# Each subject has the same blocks in the same order, but at slightly
# different places: the whole layout is shifted by a random warp of about
# ``warp_amplitude`` voxels, and the width of each block is randomly scaled by
# up to ``width_scaling``. From this subject-specific layout, we generate:
#
#   * alignment data: the responses of the blocks to the 30 conditions, plus
#     noise;
#   * a contrast map: the activated blocks, plus noise.

n_subjects, warp_amplitude, width_scaling = 10, 4.0, 0.3

alignment_data, contrast_maps, true_maps = {}, {}, {}
for i in range(n_subjects):
    widths = block_widths * rng.uniform(
        1 - width_scaling, 1 + width_scaling, size=n_blocks
    )
    blocks = block_of_voxel(
        widths, shift=warp_amplitude * rng.standard_normal()
    )

    sub = f"sub-{i + 1:02d}"
    true_maps[sub] = 0.6 * block_is_active[blocks]
    alignment_data[sub] = block_profiles[:, blocks] + rng.standard_normal(
        (n_conditions, n_voxels)
    )
    # A single contrast map, of shape (n_samples=1, n_voxels)
    contrast_maps[sub] = true_maps[sub] + noise_sd * rng.standard_normal(
        (1, n_voxels)
    )

subjects = list(contrast_maps)

# %%
# We plot the noise-free maps of two subjects: the activated blocks (shaded
# in green for the shared organization) are displaced differently in each
# subject. What we actually observe is a much noisier contrast map.

import matplotlib.pyplot as plt


def plot_maps(maps, active):
    """Plot 1D maps on top of each other, shading the ``active`` voxels."""
    fig, axes = plt.subplots(
        len(maps), 1, figsize=(8, 1.3 * len(maps)), sharex=True
    )
    for ax, (title, values) in zip(axes, maps.items(), strict=True):
        for voxel in np.flatnonzero(active):
            ax.axvspan(
                voxel - 0.5, voxel + 0.5, color="tab:green", alpha=0.2, lw=0
            )
        ax.plot(np.ravel(values), color="black", linewidth=1.2)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yticks([])
    axes[-1].set_xlabel("Voxel")
    fig.tight_layout()


plot_maps(
    {
        "Hidden truth (shared organization)": hidden_truth,
        "Noise-free map of sub-01": true_maps["sub-01"],
        "Noise-free map of sub-02": true_maps["sub-02"],
        "Observed contrast map of sub-01": contrast_maps["sub-01"],
    },
    active=hidden_truth,
)

# %%
# Predicting a subject from the others, without alignment
# -------------------------------------------------------
# The standard approach relies on anatomical correspondence: the prediction
# for a left-out subject is the voxelwise average of the other subjects'
# contrast maps. Spatial smoothing is commonly added to compensate for
# misregistration, at the price of blurring the map.

predictions = {
    "Anatomical average": {},
    "Smoothing": {},
    "Procrustes": {},
    "Optimal transport": {},
}
for sub in subjects:
    average = np.mean([contrast_maps[s] for s in subjects if s != sub], axis=0)
    predictions["Anatomical average"][sub] = average
    predictions["Smoothing"][sub] = gaussian_filter1d(average, 3, mode="wrap")

# %%
# Predicting a subject from the others, with template alignment
# -------------------------------------------------------------
# We now use :class:`~fmralign.alignment.group_alignment.GroupAlignment`
# with ``y="leave_one_subject_out"``: for each subject, a template is built
# from the alignment data of all other subjects, and the subject is aligned to
# this template. The subject never contributes to the template it is aligned
# to, exactly as a new subject would (see :ref:`sphx_glr_auto_examples_plot_template_bias.py`).
#
# We then move all contrast maps into template space. Since the templates are
# built from nearly the same subjects, they share a common space. To predict a
# left-out subject, we average the aligned maps of the other subjects, and
# send this group map back into the subject's own voxels with the transpose of
# the subject's alignment matrix ``R``: for optimal transport, it is the
# transport plan from the template to the subject; for Procrustes, it is the
# inverse rotation. Since we score predictions with a correlation, the overall
# scale of ``R`` does not matter.

from fmralign import GroupAlignment
from fmralign.methods import OptimalTransport, Procrustes

methods = {
    "Procrustes": Procrustes(),
    "Optimal transport": OptimalTransport(),
}
for name, method in methods.items():
    aligner = GroupAlignment(method=method)
    aligner.fit(alignment_data, y="leave_one_subject_out")
    aligned_maps = aligner.transform(contrast_maps)

    for sub in subjects:
        group_map = np.mean(
            [aligned_maps[s] for s in subjects if s != sub], axis=0
        )
        R = aligner.fitted_estimators[sub].R
        predictions[name][sub] = group_map @ R.T

# %%
# Comparing the predictions with the hidden truth
# -----------------------------------------------
# For each left-out subject, we measure the correlation between each
# prediction and the subject's noise-free map. We first look at the
# predictions for one subject, with its truly active voxels shaded in green.


def correlation(a, b):
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


scores = {
    name: [correlation(prediction[s], true_maps[s]) for s in subjects]
    for name, prediction in predictions.items()
}

sub = "sub-01"
plot_maps(
    {
        f"{name} with {sub} left out (correlation with hidden truth: "
        f"{correlation(prediction[sub], true_maps[sub]):.2f})": prediction[sub]
        for name, prediction in predictions.items()
    },
    active=true_maps[sub],
)

# %%
# Finally, we summarize the correlations over all left-out subjects.

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.boxplot(
    list(scores.values()),
    tick_labels=list(scores),
    orientation="horizontal",
)
ax.invert_yaxis()
ax.set_xlabel("Correlation with the hidden truth")
ax.set_title("Prediction of left-out subjects")
fig.tight_layout()
plt.show()

# %%
# Conclusion
# ----------
# Without alignment, the activated blocks of different subjects do not overlap
# well, so their average is a weak, blurred version of the truth. Smoothing
# reduces noise but blurs the map further.
#
# Template alignment matches voxels based on their responses to the
# alignment conditions rather than on their location. Procrustes does so with
# a single rotation of the whole region, which mixes the signal of distant
# voxels. Optimal transport instead moves the signal of each voxel to the
# few template locations that respond most similarly, so the activation is
# brought back to the right place while staying sharp. It therefore recovers
# the hidden truth best.
#
# You can increase ``warp_amplitude`` or ``width_scaling`` to simulate larger
# inter-subject variability: the anatomical baselines degrade quickly, while optimal
# transport remains accurate.

# %%
# References
# ----------
#
# .. footbibliography::
