# -*- coding: utf-8 -*-
# Author: Linfeng
# Date: 2025-12-18


import pandas as pd
from typing import List, Tuple

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
    figsize: Tuple[int, int] = (10, 6),
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
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = df[x_label_name]
    y = df[y_label_name]

    ax.scatter(x, y, s=point_size, alpha=alpha)

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
