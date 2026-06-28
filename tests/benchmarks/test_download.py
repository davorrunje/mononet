import hashlib
from pathlib import Path

from benchmarks.datasets.download import FILES, ZENODO_URL, sha256, verify


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"hello,world\n1,2\n")
    assert sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_verify_true_on_match_false_on_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    p.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert verify(p, digest) is True
    assert verify(p, "deadbeef") is False


def test_file_list_and_url_shape() -> None:
    assert set(FILES) == {
        f"{split}_{name}.csv"
        for split in ("train", "test")
        for name in ("auto", "blog", "compas", "heart", "loan")
    }
    assert ZENODO_URL.format(name="train_auto.csv").endswith(
        "/records/7968969/files/train_auto.csv?download=1"
    )
