"""
Robust, verified dataset download for Colab.

WHY THIS EXISTS: the original download_verified() (from your notebook)
had two bugs that combined to cause the crash you hit:

1. `subprocess` was used but never imported in the cell that defines
   it -- this only "worked" in your saved session because something
   (probably a since-deleted cell) had imported it earlier in the same
   live kernel; a fresh Runtime -> Restart and run all would hit
   NameError on the first real download.

2. The completeness check was `size_mb >= min_expected_mb` with
   min_expected_mb defaulting to 1 -- i.e. "at least 1 MB" was treated
   as "download succeeded." Your Haber Zenodo file is ~180 MB; the
   actual download stopped at ~117 MB (a dropped connection partway
   through), which still clears a 1 MB floor by a wide margin. The
   file was accepted as complete, cached, and silently reused on every
   later run -- until h5py actually tried to parse it three cells
   later and hit real structural corruption:
       OSError: Unable to synchronously open file (truncated file:
       eof = 117559032, sblock->base_addr = 0, stored_eof = 180766169)
   That "stored_eof" number IS the real file size the server was
   going to send; the download just never finished.

This version compares the downloaded byte count against the server's
own Content-Length (not a guessed MB floor), retries on failure, and
-- for .h5/.h5ad files specifically -- does one extra real check by
actually opening the file with h5py, since Content-Length isn't
guaranteed on every server. All three checks are in
tests/test_download.py, run against a local server with a real
truncated transfer (not just against a size number) -- see that file
for the ~35%-truncated-file reproduction of your exact failure mode.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests


def download_verified(
    url: str,
    out_path: str,
    min_expected_mb: float = 1,
    max_retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Download `url` to `out_path`, verifying the transfer actually
    completed. Safe to call repeatedly (e.g. rerunning a notebook) --
    a complete file already on disk is reused; an INCOMPLETE one is
    detected and re-downloaded rather than silently accepted.
    """
    expected = _remote_content_length(url, timeout=timeout)

    if os.path.exists(out_path):
        actual = os.path.getsize(out_path)
        if expected is None:
            # Can't confirm completeness against the server -- fall back
            # to the old floor, but only as a last resort.
            if actual / 1e6 >= min_expected_mb:
                print(f"{out_path} already present ({actual/1e6:.1f} MB); "
                      "server didn't report a size to verify against, trusting cache.")
                return out_path
        elif actual == expected:
            print(f"{out_path} already present and complete ({actual/1e6:.1f} MB), skipping download.")
            return out_path
        else:
            print(f"{out_path} exists but is INCOMPLETE ({actual/1e6:.1f} MB of "
                  f"{expected/1e6:.1f} MB expected) -- re-downloading, not reusing it.")
            os.remove(out_path)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                content_length = r.headers.get("Content-Length")
                content_length = int(content_length) if content_length is not None else None
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            actual = os.path.getsize(out_path)
            if content_length is not None and actual != content_length:
                raise IOError(
                    f"{out_path}: received {actual} bytes but server reported "
                    f"Content-Length {content_length} -- truncated transfer."
                )
            if actual / 1e6 < min_expected_mb:
                raise IOError(f"{out_path} looks too small ({actual/1e6:.2f} MB).")

            _validate_if_hdf5(out_path)
            print(f"Downloaded {out_path}: {actual/1e6:.1f} MB (verified complete).")
            return out_path

        except Exception as e:
            last_err = e
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt < max_retries:
                time.sleep(2 * attempt)

    raise IOError(f"Failed to download {url} to {out_path} after {max_retries} attempts: {last_err}")


def _remote_content_length(url: str, timeout: int = 60) -> Optional[int]:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def _validate_if_hdf5(path: str) -> None:
    if not (path.endswith(".h5") or path.endswith(".h5ad")):
        return
    import h5py
    try:
        with h5py.File(path, "r") as f:
            list(f.keys())  # force a real read of the file structure, not just an open()
    except Exception as e:
        raise IOError(f"{path} downloaded but is not a valid/complete HDF5 file: {e}") from e


def convert_10x_h5_to_anndata(h5_path: str, out_h5ad_path: str):
    """10x Genomics CellRanger .h5 exports (like the Breast Cancer and
    Glioblastoma Flex downloads) are a DIFFERENT format from AnnData's
    .h5ad -- neither `sc.read_h5ad()` nor cnmf's own `prepare()`
    understands the CellRanger schema (cnmf's file-type dispatch only
    recognizes .h5ad, .mtx/.mtx.gz, or delimited text -- a bare .h5
    falls through to being parsed as a text file and fails). Convert
    once with this function, then use the resulting .h5ad everywhere
    downstream (run_cnmf, run_slice, etc.).
    """
    import scanpy as sc

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    adata.write(out_h5ad_path)
    print(f"Converted {h5_path} -> {out_h5ad_path} ({adata.shape[0]} cells x {adata.shape[1]} genes)")
    return out_h5ad_path
