"""Utility helpers for removing plateau windows from time-series data."""

import numpy as np


def remove_plat(
    data_windowed: np.ndarray,
    y_pred: np.ndarray | None = None,
    return_idx: bool = False,
):
    """
    Filter out samples whose difference along the time dimension stays mostly flat.

    Parameters
    ----------
    data_windowed :
        Array shaped (N, W) containing windowed signals.
    y_pred :
        Optional array shaped (N, W) aligned with `data_windowed`. When provided,
        the same indexes are applied to `y_pred`.
    return_idx :
        When True, also return the boolean mask used for filtering.
    """
    assert len(data_windowed.shape) == 2
    data_diff = np.diff(data_windowed, axis=1)
    plat_count = np.where(data_diff == 0, 1, 0).sum(axis=1)

    cond_x = plat_count < 48
    if y_pred is None:
        filtered = data_windowed[cond_x]
        return (filtered, cond_x) if return_idx else filtered

    filtered = data_windowed[cond_x], y_pred[cond_x]
    return (*filtered, cond_x) if return_idx else filtered


__all__ = ["remove_plat"]
