"""
Single-axis comparison: overlap vs non-overlap attention for two models.
4 bars per layer (RL overlap, RL non-overlap, Base overlap, Base non-overlap).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RL_JSON = Path("/root/projects/viewsuite/rollouts/iter5_rl_960/tag_ae_loose_no_example/attn_cover_analysis/results.json")
BASE_JSON = Path("/root/projects/viewsuite/rollouts/qwen_25_vl_7b/tag_ae_loose_no_example/attn_cover_analysis/results.json")
OUTPUT = Path("/root/projects/viewsuite/rollouts/overlap_bar_compare.png")


def load_per_layer(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data["summary"]["per_layer"]


def main():
    rl = load_per_layer(RL_JSON)
    base = load_per_layer(BASE_JSON)

    layer_keys = sorted(rl.keys(), key=lambda k: int(k))
    x = np.arange(len(layer_keys))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # 4 groups: RL-overlap, RL-non-overlap, Base-overlap, Base-non-overlap
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#3B82F6", "#93C5FD", "#F97316", "#FDBA74"]
    labels = ["RL — Overlap", "RL — Non-overlap", "Base — Overlap", "Base — Non-overlap"]

    data_series = [
        ([rl[k]["overlap_mean"] for k in layer_keys], [rl[k]["overlap_std"] for k in layer_keys]),
        ([rl[k]["non_overlap_mean"] for k in layer_keys], [rl[k]["non_overlap_std"] for k in layer_keys]),
        ([base[k]["overlap_mean"] for k in layer_keys], [base[k]["overlap_std"] for k in layer_keys]),
        ([base[k]["non_overlap_mean"] for k in layer_keys], [base[k]["non_overlap_std"] for k in layer_keys]),
    ]

    for offset, color, label, (means, stds) in zip(offsets, colors, labels, data_series):
        ax.bar(x + offset, means, width, yerr=stds,
               label=label, color=color, alpha=0.85,
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Attention (log scale)", fontsize=12)
    n_trajs = rl[layer_keys[0]].get("n_trajectories", "?")
    ax.set_title(f"Overlap vs Non-overlap Attention by Layer\n"
                 f"RL (iter5_rl_960) vs Base (Qwen2.5-VL-7B)  |  n={n_trajs} trajectories",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {k}" for k in layer_keys])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
