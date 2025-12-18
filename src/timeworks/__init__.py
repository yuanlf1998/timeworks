"""Top-level package for the TimeWorks toolkit.

The package exposes a few commonly used helpers directly at import-time so
callers can simply do `from timeworks import load_data, mse` without needing
to remember the internal package layout.
"""

from importlib import metadata


def _resolve_version() -> str:
    """Return the installed package version or a sensible fallback."""
    try:
        return metadata.version("timeworks")
    except metadata.PackageNotFoundError:
        # Local development (editable installs) may not have package metadata.
        return "0.0.0"


__version__ = _resolve_version()

# Convenience re-exports for the most frequently used helpers.
from .data import load_data, norm, read_raw, dataset_config  # noqa: E402
from .metrics.error_measure import mse  # noqa: E402
from .utils.cprint import cprint  # noqa: E402

# Re-export subpackages to make them easy to discover from the root package.
from . import data, metrics, preprocessing, utils, vis, evaluate  # noqa: E402

__all__ = [
    "__version__",
    "cprint",
    "data",
    "dataset_config",
    "evaluate",
    "load_data",
    "metrics",
    "mse",
    "norm",
    "preprocessing",
    "read_raw",
    "utils",
    "vis",
]
