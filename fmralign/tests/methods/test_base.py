from fmralign.methods.base import BaseAlignment


def test_sklearn_compatibility():
    """Test basic sklearn compatibility."""
    alignment = BaseAlignment()

    # Test that it has the required sklearn interface
    assert hasattr(alignment, "fit")
    assert hasattr(alignment, "transform")
    assert hasattr(alignment, "get_params")
    assert hasattr(alignment, "set_params")
    assert hasattr(alignment, "fit_transform")
