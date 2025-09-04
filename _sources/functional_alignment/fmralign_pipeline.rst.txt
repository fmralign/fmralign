.. fmralign_pipeline:

=============================
Functional alignment pipeline
=============================

As seen in the :ref:`previous section <introduction>`,
functional alignment searches for a transform between images of two or several subjects in order to match voxels which have similar profile of activation.
This section explains how this transform is found in fmralign to make the process easy, efficient and scalable.

We compare various methods of alignment on a pairwise alignment problem for `Individual Brain Charting <https://project.inria.fr/IBC/>`_ subjects.
For each subject, we have a lot of functional information in the form of several task-based contrast per subject.
We will just work here on a ROI.


Local functional alignment
==========================

.. figure:: ../images/alignment_pipeline.png
   :scale: 25
   :align: right

Aligning images of various size is not always easy because when we search a
transformation for `n` voxels yields at least a complexity of :math:`n^2`. Moreover,
finding just one transformation for similarity of functional signal in the whole
brain could create unrealistic correspondences, for example inter-hemispheric.

To avoid these issues, we keep alignment local, i.e. on local and functionally meaningful regions.
Thus, in a first step cluster the voxels in the image into `n_pieces` sub-regions, based on functional information.
Then we find local alignment on each parcel and we recompose the global matrix from these.

With this technique, it is possible to find quickly sensible alignment even for full-brain images in 2mm resolution. The
parcellation chosen can obviously have an impact. We recommend 'ward' to have spatially compact and reproducible clusters.

.. warning::
   Optimal transport `shows poor convergence`_ for ROIs greater than 1000 voxels.
   We therefore recommend working with smaller regions when using this method.

.. _shows poor convergence: ../_images/profiling_methods.png


Alignment methods on a region
=============================

.. topic:: **Full code example on 2D simulated data**

    All the figures in this section were generated from a dedicated example:
    :ref:`sphx_glr_auto_examples_plot_alignment_simulated_2D_data.py`.

As we mentioned several times, we search for a transformation, let's call it `R`,
between the source subject data `X` and the target data `Y`. `X` and `Y` are arrays of
dimensions `(n_voxels, n_samples)` where each image is a sample.
So we can see each signal as a distribution where each voxel as a point
in a multidimensional functional space (each dimension is a sample).

We show below a 2D example, with 2 distributions: `X` in green, `Y` in red. Both have 20 voxels (points) characterized by 2 samples (images). And the alignment we search for is the matching of both distributions, optimally in some sense.

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_simulated_2D_data_001.png
   :align: left

Orthogonal alignment (Procrustes)
---------------------------------
The first idea proposed in :footcite:t:`Haxby2001` was to compute an orthogonal mixing
matrix `R` and a scaling `sc` such that Frobenius norm :math:`||sc RX - Y||^2` is minimized.

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_simulated_2D_data_003.png
   :align: left

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_simulated_2D_data_004.png
   :align: left


Optimal Transport alignment
---------------------------
Finally this package comes with a new method that build on the Wasserstein distance which is well-suited for this problem. This is the framework of Optimal Transport that search to transport all signal from `X` to `Y`
while minimizign the overall cost of this transport. `R` is here the optimal coupling between `X` and `Y` with entropic regularization.

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_simulated_2D_data_005.png
   :align: left

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_simulated_2D_data_006.png
  :align: left


Comparing those methods on a region of interest
===============================================

.. topic:: **Full code example**

    The full code example of this section is :
    :ref:`sphx_glr_auto_examples_plot_alignment_methods_benchmark.py`.

Now let's compare the performance of these various methods on our simple example:
the prediction of left-out data for a new subject from another subjects data.

Loading the data
----------------
We begin with the retrieval of images from two `Individual Brain Charting <https://project.inria.fr/IBC/>`_ subjects :

>>> from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts
>>> files, df, mask = fetch_ibc_subjects_contrasts(['sub-01', 'sub-02'])

Here `files` is the list of paths for each subject and `df` is a pandas Dataframe
with metadata about each of them.

Extract a mask for the visual cortex from Yeo Atlas
---------------------------------------------------

>>> from nilearn import datasets, plotting
>>> from nilearn.image import resample_to_img, load_img, new_img_like
>>> atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
>>> atlas = load_img(atlas_yeo_2011.thick_7)

Select visual cortex, create a mask and resample it to the right resolution

>>> mask_visual = new_img_like(atlas, atlas.get_fdata() == 1)
>>> resampled_mask_visual = resample_to_img(
    mask_visual, mask, interpolation="nearest")

Plot the mask we  use

>>> plotting.plot_roi(resampled_mask_visual, title='Visual regions mask extracted from atlas',
         cut_coords=(8, -80, 9), colorbar=True, cmap='Paired')

.. figure:: ../auto_examples/images/sphx_glr_plot_alignment_methods_benchmark_001.png
   :scale: 30
   :align: left

Define a masker
---------------
>>> from nilearn.maskers import NiftiMasker
>>> roi_masker = NiftiMasker(mask_img=mask).fit()


Prepare the data
----------------
For each subject, for each task and conditions, our dataset contains two
independent acquisitions, similar except for one acquisition parameter, the
encoding phase used that was either Antero-Posterior (AP) or Postero-Anterior (PA).
Although this induces small differences in the final data, we will take
advantage of these "duplicates" to create a training and a testing set that
contains roughly the same signals but acquired independently.


The training fold, used to learn alignment from source subject toward target:
  * source train: AP contrasts for subject 'sub-01'
  * target train: AP contrasts for subject 'sub-02'

>>> source_train = df[df.subject == 'sub-01'][df.acquisition == 'ap'].path.values
>>> target_train = df[df.subject == 'sub-02'][df.acquisition == 'ap'].path.values

The testing fold:
  * source test: PA contrasts for subject 'sub-01', used to predict
    the corresponding contrasts of subject 'sub-02'
  * target test: PA contrasts for subject 'sub-02', used as a ground truth
    to score our predictions

>>> source_test = df[df.subject == 'sub-01'][df.acquisition == 'pa'].path.values
>>> target_test = df[df.subject == 'sub-02'][df.acquisition == 'pa'].path.values

Define the estimators, fit them and do a prediction
---------------------------------------------------
To proceed with alignment we use the class PairwiseAlignment with the masker we created before.

First we choose a suitable number of regions such that each regions is approximately 100 voxels wide.

>>> n_voxels = roi_masker.mask_img_.get_fdata().sum()
>>> n_pieces = np.round(n_voxels / 100)

Then for each method we define the estimator, fit it, and predict new image. We then plot
the correlation of this prediction with the real signal. We also include identity (no alignment) as a baseline.

>>> from fmralign import GroupAlignment
>>> from fmralign.metrics import score_voxelwise
>>> methods = ["identity", "procrustes", "ot", "SRM"]
>>> titles, aligned_scores = [], []
>>> for i, method in enumerate(methods):
>>>     # Fit the group estimator on the training data
>>>     group_estimator = GroupAlignment(method=method, labels=labels).fit(
>>>         dict_source_train
>>>     )
>>>     # Compute a mapping between the template and the new subject
>>>     # using `target_train` and make a prediction using the left-out-data
>>>     target_pred = group_estimator.predict_subject(
>>>         dict_source_test, roi_masker.transform(target_train)
>>>     )
>>>     # Derive correlation between prediction, test
>>>     method_error = score_voxelwise(
>>>         target_test,
>>>         roi_masker.inverse_transform(target_pred),
>>>         masker=roi_masker,
>>>         loss="corr",
>>>     )

.. image:: ../auto_examples/images/sphx_glr_plot_alignment_methods_benchmark_002.png

We can observe that all alignment methods perform better than identity (no alignment).

References
==========

.. footbibliography::
