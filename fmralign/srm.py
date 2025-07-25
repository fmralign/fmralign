# -*- coding: utf-8 -*-
"""Test module to adapt quickly fmralign for piecewise srm
Implementation from fastSRM is taken from H. Richard
"""
# Author: T. Bazeille
# License: simplified BSD

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone

from fmralign.preprocessing import ParcellationMasker


class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        self.basis_list = [np.eye(X[0].shape[0]) for _ in range(len(X))]
        return self

    def transform(self, X, y=None, subjects_indexes=None):
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(self.basis_list))
        return np.array([X[i] for i in range(len(subjects_indexes))])

    def inverse_transform(self, X, subjects_indexes=None):
        return np.array([X for _ in subjects_indexes])

    def add_subjects(self, X_list, S):
        self.basis_list = [w for w in self.basis_list] + [
            np.eye(x.shape[0]) for x in X_list
        ]


def _get_parcel_across_subjects(parceled_data, parcel_id):
    """Get the list of parcels for a given parcel_id across subjects."""
    parcel_across_subjects = [
        parceled_data[i].to_list()[parcel_id].T
        for i in range(len(parceled_data))
    ]
    return parcel_across_subjects


def fit_one_piece(piece_X_list, method):
    """Fit SRM on group source data in one piece i, piece_X_list, using
    alignment method.

    Parameters
    ----------
    piece_X_list: ndarray
        Source data for piece i (shape : n_subjects, n_samples, n_features)
    method: string
        Algorithm used to perform groupwise alignment on piece_X_list :
        - either 'identity',
        - or an instance of IdentifiableFastSRM alignment class (imported from fastsrm)
    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fit on
    """
    if method == "identity":
        alignment_algo = Identity()
    else:
        from fastsrm.identifiable_srm import IdentifiableFastSRM

        # if isinstance(method, (FastSRM, MultiViewICA, AdaptiveMultiViewICA)):
        if isinstance(method, (IdentifiableFastSRM)):
            alignment_algo = clone(method)
            if hasattr(alignment_algo, "aggregate"):
                alignment_algo.aggregate = None
            if np.shape(piece_X_list)[1] < alignment_algo.n_components:
                alignment_algo.n_components = np.shape(piece_X_list)[1]
        else:
            warn_msg = (
                "Method not recognized, should be 'identity' "
                "or an instance of IdentifiableFastSRM"
            )
            NotImplementedError(warn_msg)

    # dirty monkey patching to avoid having n_components > n_voxels in any
    # piece which would yield a bug in add_subjects()
    alignment_algo.fit(piece_X_list)
    reduced_SR = alignment_algo.transform(piece_X_list)

    if len(reduced_SR) == len(piece_X_list):
        reduced_SR = np.mean(reduced_SR, axis=0)

    return alignment_algo, reduced_SR


class PiecewiseModel(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a SR, then use alignment to predict new contrast for target(s) subject.
    """

    def __init__(
        self,
        srm,
        n_pieces=1,
        clustering="ward",
        masker=None,
        n_jobs=1,
        verbose=0,
        reshape=True,
    ):
        """
        Parameters
        ----------
        srm : FastSRM instance
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal,
            passed to nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_pieces is then ignored.
        masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
                :class:`~nilearn.maskers.MultiNiftiMasker`, or \
                :class:`~nilearn.maskers.SurfaceMasker` , optional
            A mask to be used on the data. If provided, the mask
            will be used to extract the data. If None, a mask will
            be computed automatically with default parameters.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.srm = srm
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.masker = masker
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.reshape = reshape

    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of 4D Niimg-like or List of lists of 3D Niimg-like
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template: 4D Niimg object
            Length : n_samples

        """
        self.parcel_masker = ParcellationMasker(
            n_pieces=self.n_pieces,
            clustering=self.clustering,
            masker=self.masker,
            memory=None,
            memory_level=0,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        parceled_data = self.parcel_masker.fit_transform(imgs)
        self.masker = self.parcel_masker.masker
        self.labels_ = self.parcel_masker.labels
        self.n_pieces = self.parcel_masker.n_pieces

        outputs = Parallel(
            n_jobs=self.n_jobs, prefer="threads", verbose=self.verbose
        )(
            delayed(fit_one_piece)(
                _get_parcel_across_subjects(parceled_data, i),
                self.srm,
            )
            for i in range(self.n_pieces)
        )

        self.labels_ = self.parcel_masker.labels
        self.fit_ = [output[0] for output in outputs]
        self.reduced_sr = [output[1] for output in outputs]
        return self

    def add_subjects(self, imgs):
        """Add subject without recalculating SR"""
        for i in range(self.n_pieces):
            self.fit_[i]
            X_i = _get_parcel_across_subjects(
                self.parcel_masker.transform(imgs), i
            )
            srm = self.fit_[i]
            srm.add_subjects(X_i, self.reduced_sr[i])
        return self

    def transform(self, imgs):
        """
        Parameters
        ----------
        imgs : list of Niimgs or string (paths). Masked shape : n_voxels, n_timeframes

        Returns
        -------
        reshaped_aligned

        !!!Not implemented for n_bags>1
        """

        n_comps = self.srm.n_components
        aligned_imgs = []
        imgs_prep = self.parcel_masker.transform(imgs)
        bag_align = []
        for i, piece_srm in enumerate(self.fit_):
            X_i = [parceled_data[i].T for parceled_data in imgs_prep]
            piece_align = piece_srm.transform(X_i)
            p_comps = piece_srm.n_components
            if p_comps != n_comps:
                piece_align = [
                    np.pad(
                        t,
                        ((0, n_comps - p_comps), (0, 0)),
                        mode="constant",
                    )
                    for t in piece_align
                ]
            bag_align.append(piece_align)
        aligned_imgs.append(bag_align)
        reordered_aligned = np.moveaxis(aligned_imgs[0], [1], [0])
        if self.reshape is False:
            return reordered_aligned
        reshaped_aligned = reordered_aligned.reshape(
            len(imgs), -1, np.shape(aligned_imgs)[-1]
        )
        return reshaped_aligned

    def get_full_basis(self):
        """
        Concatenated (and padded if needed) local basis for each subject to
        fullbrain basis of shape (n_components,n_voxels) for each subject
        """
        unique_labels = np.unique(self.labels_[0])
        n_comps = self.srm.n_components
        n_subs = len(self.fit_[0].basis_list)
        full_basis_list = np.zeros(shape=(n_subs, len(self.labels_), n_comps))

        for k in range(len(unique_labels)):
            label = unique_labels[k]
            k_estim = self.fit_[k]
            i = label == self.labels_
            p_comps = k_estim.n_components
            for s, basis in enumerate(k_estim.basis_list):
                full_basis_list[s, i, :] = np.pad(
                    basis, ((0, n_comps - p_comps), (0, 0)), mode="constant"
                )

        basis_imgs = [
            self.masker.inverse_transform(full_basis.T)
            for full_basis in full_basis_list
        ]
        return basis_imgs

    def fit_transform(self, imgs):
        return self.fit(imgs).transform(imgs)

    def inverse_transform(self, X_transformed):
        # TODO: Just inverse operations in transform to make API compliant
        pass

    def clean(self):
        """
        Just clean temporary directories
        """
        for srm in self.fit_:
            srm.clean()
