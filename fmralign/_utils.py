# -*- coding: utf-8 -*-
import datetime
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from nilearn.image import new_img_like
from nilearn.masking import intersect_masks
from sklearn.exceptions import NotFittedError


def _intersect_clustering_mask(clustering, mask):
    """Take 3D Niimg clustering and bigger mask, output reduced mask."""
    dat = clustering.get_fdata()
    new_ = np.zeros_like(dat)
    new_[dat > 0] = 1
    clustering_mask = new_img_like(clustering, new_)
    return intersect_masks(
        [clustering_mask, mask], threshold=1, connected=True
    )


def save_alignment(alignment_estimator, output_path):
    """Save the alignment estimator object to a file.

    Parameters
    ----------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The alignment estimator object to be saved.
        It should be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`.
        The object should have been fitted before saving.
    output_path : str or Path
        Path to the file or directory where the model will be saved.
        If a directory is provided, the model will be saved with a
        timestamped filename in that directory.
        If a file is provided, the model will be saved with that filename.

    Raises
    ------
    NotFittedError
        If the alignment estimator has not been fitted yet.
    ValueError
        If the output path is not a valid file or directory.
    """
    if not hasattr(alignment_estimator, "fitted_estimators"):
        raise NotFittedError(
            "This instance has not been fitted yet. "
            "Please call 'fit' before 'save'."
        )

    output_path = Path(output_path)

    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"alignment_estimator_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        joblib.dump(alignment_estimator, output_path / suffix)

    else:
        joblib.dump(alignment_estimator, output_path)


def load_alignment(input_path):
    """Load an alignment estimator object from a file.

    Parameters
    ----------
    input_path : str or Path
        Path to the file or directory from which the model will be loaded.
        If a directory is provided, the latest .pkl file in that directory
        will be loaded.

    Returns
    -------
    alignment_estimator : :obj:`PairwiseAlignment` or :obj:`TemplateAlignment`
        The loaded alignment estimator object.
        It will be an instance of either `PairwiseAlignment` or
        `TemplateAlignment`, depending on what was saved.

    Raises
    ------
    ValueError
        If no .pkl files are found in the directory or if the input path is not
        a valid file or directory.
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # If it's a directory, look for the latest .pkl file
        pkl_files = list(input_path.glob("*.pkl"))
        if not pkl_files:
            raise ValueError(
                f"No .pkl files found in the directory: {input_path}"
            )
        input_path = max(pkl_files, key=lambda x: x.stat().st_mtime)

    return joblib.load(input_path)
