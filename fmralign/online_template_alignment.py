"""Module for sparse template alignment."""

import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import (
    get_modality_features,
    _sparse_clusters_radius,
)
from fmralign.alignment_methods import SparseUOT
from nilearn.masking import apply_mask_fmri, unmask
from nilearn._utils.cache_mixin import _check_memory
from functools import partial


class SubjectDataset(Dataset):
    def __init__(self, imgs, subject_loader):
        self.imgs = imgs
        self.subject_loader = subject_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        return self.subject_loader(img_path)


def align_to_template(
    imgs,
    template_data,
    sparsity_mask,
    subject_loader,
    verbose=False,
    device="cpu",
    **kwargs,
):
    alignment_estimators = []
    for img in imgs:
        img_data = subject_loader(img)
        estimator = SparseUOT(
            sparsity_mask,
            method="sinkhorn",
            device=device,
            verbose=max(0, verbose - 1),
            **kwargs,
        )
        estimator.fit(img_data, template_data)
        alignment_estimators.append(estimator)

    return alignment_estimators


def load_one_subject(imgs, parcellation_img, masker, modality):
    imgs_ = get_modality_features(imgs, parcellation_img, masker, modality)
    return apply_mask_fmri(imgs_, masker.mask_img_).astype(np.float32)


def fit_online_template(
    imgs,
    sparsity_mask,
    subject_loader,
    n_iter=100,
    verbose=False,
    device="cpu",
    num_workers=2,
    prefetch_factor=2,
    **kwargs,
):
    template_data = subject_loader(imgs[0])

    n_iter_ = max(n_iter, len(imgs))
    # Create a dataset and a data loader
    dataset = SubjectDataset(imgs, subject_loader)
    # The sampler ensures we sample with replacement for n_iter steps
    sampler = RandomSampler(dataset, replacement=True, num_samples=n_iter_)

    # The DataLoader will use worker processes to call dataset.__getitem__ in the background
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,  # Each batch contains one subject's data
        num_workers=num_workers,  # Number of CPU processes to load data in parallel
        prefetch_factor=prefetch_factor,  # Number of batches to prefetch per worker
        pin_memory=False,  # Not needed for CPU-only workflow
    )

    estimator = SparseUOT(
        sparsity_mask,
        method="sinkhorn_divergence",
        device=device,
        verbose=max(0, verbose - 1),
        **kwargs,
    )

    # Main training loop
    for i, img_data_batch in enumerate(dataloader):
        if verbose:
            print(f"Iteration {i + 1}/{n_iter}")

        current_img_data = img_data_batch.squeeze(0).numpy()
        estimator.fit(template_data, current_img_data)
        alpha = 1 / (i + 2)
        template_data = (
            1 - alpha
        ) * template_data + alpha * estimator.transform(template_data)

    return template_data


class OnlineTemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a template, then use pairwise alignment to predict
    new contrast for target subject.
    """

    def __init__(
        self,
        n_pieces=1,
        clustering="kmeans",
        n_iter=2,
        save_template=None,
        masker=None,
        radius=5,
        modality="response",
        device="cpu",
        n_jobs=1,
        verbose=0,
        memory=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
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
        n_iter: int
            number of iteration in the alternate minimization. Each img is
            aligned n_iter times to the evolving template. If n_iter = 0,
            the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
        masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
                :class:`~nilearn.maskers.MultiNiftiMasker`, or \
                :class:`~nilearn.maskers.SurfaceMasker` , optional
            A mask to be used on the data. If provided, the mask
            will be used to extract the data. If None, a mask will
            be computed automatically with default parameters.
        radius: int, optional (default = 5)
            Radius in mm to define the neighborhood for each voxel.
        modality : str, optional (default='response')
            Specifies the alignment modality to be used:
            * 'response': Aligns by directly comparing corresponding similar 
            time points in the source and target images.
            * 'connectivity': Aligns based on voxel-wise connectivity features 
            within each parcel, comparing how each voxel relates to others in 
            the same region.
            * 'hybrid': Combines both time series and connectivity information 
            to perform the alignment.
        device: string, optional (default = 'cpu')
            Device on which the computation will be done. If 'cuda', the
            computation will be done on the GPU if available.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.

        """
        self.template = None
        self.template_history = None
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.n_iter = n_iter
        self.save_template = save_template
        self.masker = masker
        self.radius = radius
        self.modality = modality
        self.device = device
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose
        self.memory = memory
        self.kwargs = kwargs

    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of `str` paths.
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
        self.memory = _check_memory(self.memory)
        if self.memory is not None:
            subject_loader = self.memory.cache(
                partial(
                    load_one_subject,
                    parcellation_img=self.clustering,
                    masker=self.masker,
                    modality=self.modality,
                )
            )
        else:
            subject_loader = partial(
                load_one_subject,
                parcellation_img=self.clustering,
                masker=self.masker,
                modality=self.modality,
            )

        self.sparsity_mask = _sparse_clusters_radius(
            self.masker.mask_img_, self.radius
        )

        template_data = fit_online_template(
            imgs=imgs,
            sparsity_mask=self.sparsity_mask,
            subject_loader=subject_loader,
            n_iter=self.n_iter,
            verbose=max(0, self.verbose - 1),
            device=self.device,
            **self.kwargs,
        )

        self.template = unmask(template_data, self.masker.mask_img_)
        self.fit_ = align_to_template(
            imgs=imgs,
            template_data=template_data,
            sparsity_mask=self.sparsity_mask,
            subject_loader=subject_loader,
            device=self.device,
            verbose=max(0, self.verbose - 1),
            **self.kwargs,
        )

        if self.save_template is not None:
            self.template.to_filename(self.save_template)

    def transform(self, img, subject_index=None):
        """
        Transform a (new) subject image into the template space.

        Parameters
        ----------
        img: 4D Niimg-like object
            Subject image.
        subject_index: int, optional (default = None)
            Index of the subject to be transformed. It should
            correspond to the index of the subject in the list of
            subjects used to fit the template. If None, a new
            `PairwiseAlignment` object is fitted between the new
            subject and the template before transforming.


        Returns
        -------
        predicted_imgs: 4D Niimg object
            Transformed data.

        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        if subject_index is None:
            raise NotImplementedError(
                "Transforming a new subject without specifying "
                "the subject index is not implemented yet. "
                "Please specify the subject index."
            )
        else:
            alignment_estimator = self.fit_[subject_index]
            img_data = self.masker.transform(img)
            X_transformed = alignment_estimator.transform(img_data)
            return self.masker.inverse_transform(X_transformed)

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.labels
            parcellation_img = self.clustering
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
