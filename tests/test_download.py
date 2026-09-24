"""
Tests for slice_lca.data_download, using mocked HTTP responses so the
suite doesn't depend on network access. The h5py-corruption-detection
path IS tested against a real (deliberately truncated) HDF5 file,
since that's the actual backstop that would have caught the failure
mode this module exists to fix.
"""
import os
from unittest.mock import patch, MagicMock

import h5py
import pytest

from slice_lca.data_download import download_verified, _validate_if_hdf5


def _mock_response(content: bytes, content_length: int = None, status_ok=True):
    resp = MagicMock()
    resp.headers = {"Content-Length": str(content_length if content_length is not None else len(content))}
    resp.raise_for_status = MagicMock() if status_ok else MagicMock(side_effect=Exception("HTTP error"))
    resp.iter_content = MagicMock(return_value=[content])
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_fresh_complete_download(tmp_path):
    out = tmp_path / "file.bin"
    content = b"x" * 2_000_000  # 2 MB
    with patch("slice_lca.data_download.requests.head") as mock_head, \
         patch("slice_lca.data_download.requests.get") as mock_get:
        mock_head.return_value = MagicMock(headers={"Content-Length": str(len(content))})
        mock_get.return_value = _mock_response(content)
        download_verified("http://example.com/file.bin", str(out), min_expected_mb=0.1)
    assert out.read_bytes() == content


def test_truncated_transfer_is_detected_and_retried_then_fails(tmp_path):
    """The core bug: a transfer that stops short of Content-Length must
    be rejected, not accepted because it clears an arbitrary MB floor."""
    out = tmp_path / "file.bin"
    full_content = b"x" * 2_000_000
    truncated_content = full_content[: int(len(full_content) * 0.6)]  # stops short

    with patch("slice_lca.data_download.requests.head") as mock_head, \
         patch("slice_lca.data_download.requests.get") as mock_get:
        mock_head.return_value = MagicMock(headers={"Content-Length": str(len(full_content))})
        # Server claims Content-Length = full size but only sends the truncated bytes,
        # exactly like the real Zenodo failure (stored_eof > actual eof).
        mock_get.return_value = _mock_response(truncated_content, content_length=len(full_content))
        with pytest.raises(IOError, match="truncated"):
            download_verified("http://example.com/file.bin", str(out), min_expected_mb=0.1, max_retries=1)
    # Must NOT leave the truncated file sitting on disk masquerading as complete.
    assert not out.exists()


def test_stale_truncated_cache_is_redownloaded_not_reused(tmp_path):
    """THE ACTUAL BUG from the notebook: a previously-truncated file
    already on disk must be detected and replaced, not silently reused
    just because it clears the size floor."""
    out = tmp_path / "file.bin"
    full_content = b"x" * 2_000_000
    out.write_bytes(full_content[:1_200_000])  # 1.2 MB stale truncated file, clears old 1MB floor

    with patch("slice_lca.data_download.requests.head") as mock_head, \
         patch("slice_lca.data_download.requests.get") as mock_get:
        mock_head.return_value = MagicMock(headers={"Content-Length": str(len(full_content))})
        mock_get.return_value = _mock_response(full_content)
        download_verified("http://example.com/file.bin", str(out), min_expected_mb=0.1)

    assert out.read_bytes() == full_content
    mock_get.assert_called_once()  # confirms it actually re-downloaded, didn't just trust the cache


def test_complete_cache_is_not_redownloaded(tmp_path):
    out = tmp_path / "file.bin"
    full_content = b"x" * 2_000_000
    out.write_bytes(full_content)

    with patch("slice_lca.data_download.requests.head") as mock_head, \
         patch("slice_lca.data_download.requests.get") as mock_get:
        mock_head.return_value = MagicMock(headers={"Content-Length": str(len(full_content))})
        download_verified("http://example.com/file.bin", str(out), min_expected_mb=0.1)

    mock_get.assert_not_called()


def test_hdf5_validation_catches_truncated_file_even_with_matching_length(tmp_path):
    """Backstop for servers that don't send Content-Length: build a
    REAL valid HDF5 file, truncate it for real, and confirm the
    explicit h5py-open check (not just a byte count) catches it --
    this reproduces the user's exact OSError."""
    complete = tmp_path / "complete.h5"
    with h5py.File(complete, "w") as f:
        f.create_dataset("data", data=list(range(100_000)))

    truncated = tmp_path / "truncated.h5"
    data = complete.read_bytes()
    truncated.write_bytes(data[: int(len(data) * 0.65)])

    with pytest.raises(IOError, match="not a valid/complete HDF5"):
        _validate_if_hdf5(str(truncated))

    # And confirm a genuinely complete file passes.
    _validate_if_hdf5(str(complete))
