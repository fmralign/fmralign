# -*- coding: utf-8 -*-

import numpy as np

from numpy.testing import assert_array_almost_equal
from scipy.sparse import csc_matrix

from fmralign.methods.alignment_methods import (
    DiagonalAlignment,
    Identity,
    OptimalTransport,
    RidgeAlignment,
    ScaledOrthogonal,
)
from fmralign.tests.utils import zero_mean_coefficient_determination


def test_all_classes_R_and_pred_shape_and_better_than_identity():
    """Test all classes on random case"""
    # test on empty data
    X = np.zeros((30, 10))
    for algo in [
        Identity(),
        RidgeAlignment(),
        ScaledOrthogonal(),
        OptimalTransport(),
        DiagonalAlignment(),
    ]:
        algo.fit(X, X)
        assert_array_almost_equal(algo.transform(X), X)
    # if trying to learn a fit from array of zeros to zeros (empty parcel)
    # every algo will return a zero matrix
    for n_samples, n_features in [(100, 20), (20, 100)]:
        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples, n_features)
        id = Identity()
        id.fit(X, Y)
        identity_baseline_score = zero_mean_coefficient_determination(Y, X)
        assert_array_almost_equal(X, id.transform(X))
        for algo in [
            RidgeAlignment(),
            ScaledOrthogonal(),
            ScaledOrthogonal(scaling=False),
            OptimalTransport(),
            DiagonalAlignment(),
        ]:
            algo.fit(X, Y)
            # test that permutation matrix shape is (20, 20) except for Ridge
            if isinstance(algo.R, csc_matrix):
                R = algo.R.toarray()
                assert R.shape == (n_features, n_features)
            elif not isinstance(algo, RidgeAlignment):
                R = algo.R
                assert R.shape == (n_features, n_features)
            # test pred shape and loss improvement compared to identity
            X_pred = algo.transform(X)
            assert X_pred.shape == X.shape
            algo_score = zero_mean_coefficient_determination(Y, X_pred)
            assert algo_score >= identity_baseline_score
