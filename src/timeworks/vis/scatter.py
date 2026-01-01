# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-18


from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt


def lists_to_df(dataset_names: List[str], **metrics: List[float]) -> pd.DataFrame:
    """
    Convert multiple metric lists into a DataFrame.

    Parameters
    ----------
    dataset_names : List[str]
        Dataset identifiers (used for labeling).
    **metrics : List[float]
        Named metric lists, e.g. x=[...], y=[...], k_model=[...]

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per dataset.
    """
    n = len(dataset_names)
    for k, v in metrics.items():
        if len(v) != n:
            raise ValueError(f"Length mismatch: {k} has {len(v)} entries, expected {n}")

    data = {"dataset": dataset_names}
    data.update(metrics)

    return pd.DataFrame(data)


def scatter_from_df(
    df: pd.DataFrame,
    *,
    x_label_name: str,
    y_label_name: str,
    dataset_name: str = "dataset",
    colors: List[float] | None = None,
    cmap: str = "coolwarm",
    figsize: Tuple[int, int] = (10, 6),
    ax: plt.Axes | None = None,
    annotate: bool = True,
    point_size: int = 100,
    alpha: float = 0.7,
):
    """
    Scatter plot using columns from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_label_name : str
        Column name used for x-axis.
    y_label_name : str
        Column name used for y-axis.
    dataset_name : str
        Column name used for point labels.
    colors : List[float] | None
        Optional color values aligned with `dataset_name`.
    cmap : str
        Matplotlib colormap name used when `colors` is provided.
    ax : plt.Axes | None
        Optional axes to draw on. When omitted, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = df[x_label_name]
    y = df[y_label_name]

    if colors is None:
        scatter = ax.scatter(x, y, s=point_size, alpha=alpha)
    else:
        scatter = ax.scatter(x, y, s=point_size, alpha=alpha, c=colors, cmap=cmap)
        fig.colorbar(scatter, ax=ax)

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                str(row[dataset_name]),
                (row[x_label_name], row[y_label_name]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

    ax.set_xlabel(x_label_name)
    ax.set_ylabel(y_label_name)
    ax.set_title(f"{x_label_name} vs {y_label_name}")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import pearsonr, spearmanr


def scatter_with_regression(
    x: List[float] | np.ndarray,
    y: List[float] | np.ndarray,
    *,
    ax: plt.Axes | None = None,
    figsize: Tuple[int, int] = (6, 4),
    point_size: int = 12,
    alpha: float = 0.7,
    color: str = "tab:blue",
    line_color: str = "#2e3f8f",
    line_style: str = "--",
    line_width: float = 2.5,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    rankic_method: str = "pearson",
    show_rankic_in_title: bool = True,
    grid_alpha: float = 0.3,
    # ⭐ 新增：是否启用 outlier clipping
    clip_outliers: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    """
    Scatter plot with optional regression line and RankIC.
    Supports optional percentile-based outlier clipping.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.asarray(x)
    y = np.asarray(y)

    # ----- 1️⃣ nan-safe -----
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # ----- 2️⃣ ⭐ 可选 outlier clipping -----
    if clip_outliers and len(x) > 10:

        # x 方向
        x_lo, x_hi = np.quantile(x, [q_low, q_high])
        mask_x = (x >= x_lo) & (x <= x_hi)

        # y 方向
        y_lo, y_hi = np.quantile(y, [q_low, q_high])
        mask_y = (y >= y_lo) & (y <= y_hi)

        # 共同保留
        mask = mask_x & mask_y
        x = x[mask]
        y = y[mask]

    # ----- 3️⃣ 画散点 -----
    ax.scatter(x, y, s=point_size, alpha=alpha, color=color)

    # ----- 4️⃣ 拟合 + RankIC -----
    if len(x) >= 2:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(
            xs,
            a * xs + b,
            line_style,
            color=line_color,
            linewidth=line_width,
            alpha=0.95,
            zorder=5,
        )

        if rankic_method == "spearman":
            rho, _ = spearmanr(x, y)
        else:
            rho, _ = pearsonr(x, y)
        rho_str = f"{rho:.3f}"
    else:
        rho = float("nan")
        rho_str = "nan"

    # ----- 5️⃣ 标题 -----
    if show_rankic_in_title:
        if title is None:
            title = f"RankIC={rho_str}"
        else:
            title = f"{title}  (RankIC={rho_str})"

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    ax.grid(alpha=grid_alpha)
    fig.tight_layout()

    return fig, ax, rho

if __name__ == '__main__':
    df = lists_to_df(
    dataset_names=['ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Electricity'],
    mse_actual = [0.3757, 0.2914, 0.3209, 0.1778, 0.4464, 0.1737], 
    k_dataset = [5.5686, 6.4254, 10.8808, 6.5909, 13.8148, 5.4735],
    )

    fig, ax = scatter_from_df(
    df,
    x_label_name="mse_actual",
    y_label_name="k_dataset",
    )
    plt.show()
