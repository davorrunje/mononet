"""Basic tests for mononet."""

import mononet


def test_version_is_set() -> None:
    assert isinstance(mononet.__version__, str)
    assert mononet.__version__ != ""
