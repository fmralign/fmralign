import numpy as np

from fmralign.embeddings.connectivity import get_connectivity_features


def test_get_connectivity_features():
    """Test the validity of connectivity features."""
    n_voxels = 2
    n_samples = 100
    labels = np.array([1, 2])

    data = np.vstack(
        [
            np.random.rand(n_samples),
            np.random.randn(n_samples),
        ]
    ).T
    res = get_connectivity_features(data, labels)

    assert res.shape == (len(np.unique(labels)), n_voxels)
    assert np.all(np.isfinite(res))
    assert np.all(res >= -(1 + 1e-6)) and np.all(res <= 1 + 1e-6)
