from .core import (
    SLICEResult,
    fit_slice,
    sparse_svd,
    sparse_svd_multistart,
    canonicalize_signs,
    compute_kME,
    assign_modules,
    select_k_parallel_analysis,
    sparsity_budget,
    bh_fdr,
    NotPreprocessedError,
)
from .data_download import download_verified, convert_10x_h5_to_anndata
from . import preprocessing

__all__ = [
    "SLICEResult",
    "fit_slice",
    "sparse_svd",
    "sparse_svd_multistart",
    "canonicalize_signs",
    "compute_kME",
    "assign_modules",
    "select_k_parallel_analysis",
    "sparsity_budget",
    "bh_fdr",
    "NotPreprocessedError",
    "download_verified",
    "convert_10x_h5_to_anndata",
    "preprocessing",
]

__version__ = "0.1.0"
