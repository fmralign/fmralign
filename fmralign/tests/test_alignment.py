from fmralign.alignment import GroupAlignment
import numpy as np
from fmralign.methods import Identity, ScaledOrthogonalAlignment, SparseUOT


def test_alignment():
    # Create a mock dataset
    n_subjects = 3
    n_features = 10
    n_voxels = 30

    X = [np.random.rand(n_features, n_voxels) for _ in range(n_subjects)]

    method = SparseUOT()
    algo = GroupAlignment(
        method=method,
        target=None,
        labels=np.random.randint(1, 3, n_voxels),
        verbose=10,
    )
    algo.fit(X)
    transformed = algo.transform(X, subject_indices=[0, 1, 2])

    assert len(transformed) == n_subjects


test_alignment()
