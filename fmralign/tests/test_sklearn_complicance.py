from sklearn.utils.estimator_checks import parametrize_with_checks

from fmralign import GroupAlignment, PairwiseAlignment
from fmralign.methods import (
    DetSRM,
    Identity,
    OptimalTransport,
    Procrustes,
    RidgeAlignment,
    # SpectralOT
)
from fmralign.methods.base import BaseAlignment

ESTIMATORS_TO_CHECK = [
    PairwiseAlignment(),
    GroupAlignment(),
    DetSRM(),
    Identity(),
    OptimalTransport(),
    Procrustes(),
    RidgeAlignment(),
    # SpectralOT(evecs=np.ones((5,5))),
]


def return_expected_failed_checks(
    estimator,
) -> dict[str, str]:
    """Return the expected failures for a given estimator.

    This will say which of the sklearn checks are expected to fail
    for a given nilearn estimator,
    with the reason why or saying what home made check replaces it.

    Returns
    -------
    expected_failed_checks : dict[str, str]
        A dictionary of the form::

            {
                "check_name": "this check is expected to fail because ...",
            }

        Where `"check_name"` is the name of the check, and `"my reason"` is why
        the check fails.
    """
    expected_failed_checks: dict[str, str] = {}

    expected_failed_checks = {
        "check_n_features_in_after_fitting": "TODO",
        "check_n_features_in": "TODO",
        "check_dict_unchanged": "TODO",
        "check_dont_overwrite_parameters": "TODO",
        "check_methods_subset_invariance": "TODO",
        "check_estimator_sparse_tag": "TODO",
        "check_fit_check_is_fitted": "TODO",
        "check_estimator_sparse_matrix": "TODO",
        "check_pipeline_consistency": "TODO",
        "check_complex_data": "TODO",
        "check_fit_score_takes_y": "TODO",
        "check_estimators_pickle": "TODO",
        "check_estimators_nan_inf": "TODO",
        "check_fit_idempotent": "TODO",
        "check_f_contiguous_array_estimator": "TODO",
        "check_dtype_object": "TODO",
        "check_methods_sample_order_invariance": "TODO",
        "check_estimators_dtypes": "TODO",
        "check_positive_only_tag_during_fit": "TODO",
        "check_no_attributes_set_in_init": "TODO",
        "check_estimator_sparse_array": "TODO",
        "check_readonly_memmap_input": "TODO",
        "check_estimators_fit_returns_self": "TODO",
        "check_estimators_overwrite_params": "TODO",
        "check_estimators_empty_data_messages": "TODO",
    }

    if isinstance(estimator, RidgeAlignment):
        expected_failed_checks |= {}

    if isinstance(estimator, OptimalTransport):
        expected_failed_checks |= {
            "check_transformer_n_iter": "TODO",
            "check_do_not_raise_errors_in_init_or_set_params": "TODO",
            "check_fit1d": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_fit2d_1sample": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

    if isinstance(estimator, (Identity,)):
        expected_failed_checks |= {
            "check_fit1d": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformers_unfitted": "TODO",
        }

    if isinstance(estimator, (Procrustes)):
        expected_failed_checks |= {
            "check_fit1d": "TODO",
            "check_fit2d_1sample": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

    if isinstance(estimator, (DetSRM)):
        expected_failed_checks |= {
            "check_fit1d": "TODO",
            "check_fit2d_1sample": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
            "check_transformer_data_not_an_array": "TODO",
        }

    if isinstance(estimator, (PairwiseAlignment)):
        expected_failed_checks |= {
            "check_fit1d": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

    if isinstance(estimator, (GroupAlignment)):
        expected_failed_checks |= {
            "check_fit1d": "TODO",
            "check_fit2d_1sample": "TODO",
            "check_fit2d_1feature": "TODO",
            "check_fit2d_predict1d": "TODO",
            "check_transformer_general": "TODO",
            "check_transformer_data_not_an_array": "TODO",
            "check_transformer_preserve_dtypes": "TODO",
        }

    return expected_failed_checks


@parametrize_with_checks(
    estimators=ESTIMATORS_TO_CHECK,
    expected_failed_checks=return_expected_failed_checks,
)
def test_check_estimator_sklearn(estimator, check):
    """Check compliance with sklearn estimators."""
    check(estimator)
