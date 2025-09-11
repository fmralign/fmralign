import numpy as np
from numpy.testing import assert_array_almost_equal

from fmralign import metrics


def test_score_voxelwise():
    A = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.2, 0.2, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.0, 0.2, -1.0],
            [0.2, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.2, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
    )

    # check correlation raw_values
    correlation1 = metrics.score_voxelwise(A, B, loss="corr")
    assert_array_almost_equal(correlation1, [1.0, -0.25, -1])

    # check correlation uniform_average
    correlation2 = metrics.score_voxelwise(
        A, B, loss="corr", multioutput="uniform_average"
    )
    assert correlation2.ndim == 0

    # check R2
    r2 = metrics.score_voxelwise(A, B, loss="R2")
    assert_array_almost_equal(r2, [-1.0, -1.0, -1.0])

    # check normalized reconstruction
    norm_rec = metrics.score_voxelwise(A, B, loss="n_reconstruction_err")
    assert_array_almost_equal(norm_rec, [0.14966, 0.683168, -1.0])


def test_normalized_reconstruction_error():
    A = np.asarray([[1, 1.2, 1, 1.2, 1], [1, 1, 1, 0.2, 1], [1, -1, 1, -1, 1]])
    B = np.asarray(
        [[0, 0.2, 0, 0.2, 0], [0.2, 1, 1, 1, 1], [-1, 1, -1, 1, -1]]
    )

    avg_norm_rec = metrics.normalized_reconstruction_error(
        A, B, multioutput="uniform_average"
    )
    np.testing.assert_almost_equal(avg_norm_rec, -0.788203)
