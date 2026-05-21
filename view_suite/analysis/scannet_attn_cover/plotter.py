"""
Bar plot: overlap vs non-overlap mean attention per layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_FIG_SIZE = (10, 6)
_DPI = 150
_COLOR_OVERLAP = "#3B82F6"       # blue
_COLOR_NON_OVERLAP = "#F97316"   # orange


def plot_overlap_bar(
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate bar plots comparing overlap vs non-overlap mean attention.

    Creates:
      - Per-layer individual bar plots (2 bars each).
      - A combined plot with all layers side by side.
    """
    output_dir = Path(output_dir)
    per_layer = summary["per_layer"]
    layer_keys = sorted(per_layer.keys(), key=lambda k: int(k))

    # --- Per-layer plots ---
    for lk in layer_keys:
        stats = per_layer[lk]
        _plot_single_layer(stats, int(lk), output_dir)

    # --- Combined plot ---
    if len(layer_keys) >= 2:
        _plot_combined(per_layer, layer_keys, output_dir)


def _plot_single_layer(stats: Dict, layer_idx: int, output_dir: Path) -> None:
    """Two-bar plot for a single layer."""
    fig, ax = plt.subplots(figsize=(5, 5))

    means = [stats["overlap_mean"], stats["non_overlap_mean"]]
    stds = [stats["overlap_std"], stats["non_overlap_std"]]
    colors = [_COLOR_OVERLAP, _COLOR_NON_OVERLAP]
    labels = ["Overlap\n(any 2+ views)", "Non-overlap"]

    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=6, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Attention", fontsize=12)
    ax.set_title(f"Layer {layer_idx} — Overlap vs Non-overlap Attention\n"
                 f"(n={stats['n_trajectories']} trajectories)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.0001,
                f"{m:.5f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    path = output_dir / f"overlap_bar_layer_{layer_idx}.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_combined(
    per_layer: Dict[str, Dict],
    layer_keys: list,
    output_dir: Path,
) -> None:
    """Grouped bar plot with all layers."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    x = np.arange(len(layer_keys))
    width = 0.35

    overlap_means = [per_layer[k]["overlap_mean"] for k in layer_keys]
    overlap_stds = [per_layer[k]["overlap_std"] for k in layer_keys]
    non_overlap_means = [per_layer[k]["non_overlap_mean"] for k in layer_keys]
    non_overlap_stds = [per_layer[k]["non_overlap_std"] for k in layer_keys]

    bars1 = ax.bar(x - width / 2, overlap_means, width, yerr=overlap_stds,
                   label="Overlap (any 2+ views)", color=_COLOR_OVERLAP,
                   alpha=0.85, capsize=4, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, non_overlap_means, width, yerr=non_overlap_stds,
                   label="Non-overlap", color=_COLOR_NON_OVERLAP,
                   alpha=0.85, capsize=4, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Attention", fontsize=12)
    n_trajs = per_layer[layer_keys[0]].get("n_trajectories", "?")
    ax.set_title(f"Overlap vs Non-overlap Attention by Layer\n"
                 f"(n={n_trajs} trajectories)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {k}" for k in layer_keys])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "overlap_bar_combined.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")
