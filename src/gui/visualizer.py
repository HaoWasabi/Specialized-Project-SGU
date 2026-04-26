from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Ma trận tương quan giữa các biến và giá vàng",
    figsize: tuple[int, int] = (9, 7),
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)

    for row_index in range(corr_df.shape[0]):
        for column_index in range(corr_df.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{corr_df.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Hệ số tương quan")
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_prediction_comparison(
    y_true: np.ndarray,
    predictions: np.ndarray,
    title: str,
    true_label: str,
    prediction_label: str,
    figsize: tuple[int, int] = (15, 7),
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_true, label=true_label, color="blue", linewidth=2)
    ax.plot(predictions, label=prediction_label, color="red", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Thời gian (ngày trong tập test)")
    ax.set_ylabel("Giá vàng (USD)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax