import numpy as np
from fmralign.template.utils import (
    _fit_sparse_template,
    _reconstruct_template,
    _fit_local_template,
)
from joblib import Parallel, delayed


def check_template_method(method, piecewise=False):
    pass


def fit_template(X, method, labels, n_jobs, verbose):
    n_labels = len(np.unique(labels))
    if n_labels == 1:
        # If only one label, use the whole brain method
        check_template_method(method, piecewise=False)
        return fit_template_whole_brain(X, method, labels, n_jobs, verbose)
    else:
        # If multiple labels, use the piecewise method
        check_template_method(method, piecewise=True)
        return fit_template_piecewise(X, method, labels, n_jobs, verbose)


def fit_template_piecewise(X, method, labels, n_jobs, verbose):
    fitted_estimators = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(_fit_local_template)(
            parcel_i,
            n_iter,
            scale_template,
            method,
        )
        for parcel_i in parcels_data
    )
    template, template_history = _reconstruct_template(fit_, labels, masker)
    return fitted_estimators, template


def fit_template_whole_brain(X, method, labels, n_jobs, verbose):
    template, fitted_estimators = _fit_sparse_template(
        subjects_data=X,
        sparsity_mask=sparsity_mask,
        n_iter=n_iter,
        scale_template=scale_template,
        alignment_method=method,
        device=device,
        verbose=True if verbose > 0 else False,
        **kwargs,
    )
    return fitted_estimators, template
