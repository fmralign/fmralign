from fmralign.alignment import Alignment
import numpy as np


def test_alignment():
    # Create a mock dataset
    n_subjects = 3
    n_features = 10
    n_voxels = 30

    X = [np.random.rand(n_features, n_voxels) for _ in range(n_subjects)]
    algo = Alignment(
        method="identity",
        target=None,
        labels=np.ones(n_voxels),
    )
    algo.fit(X)
    transformed = algo.transform(X, subject_indices=[0, 1, 2])

    assert len(transformed) == n_subjects


test_alignment()
