import pytest


def test_import() -> None:
    try:
        import mononet
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")
    else:
        assert mononet is not None, "Module imported but is None"
