"""
Compare two models' full-sequence attention-to-image results.

Plots:
  1. Per-turn response→image fraction by turn index (line plot, both models).
  2. Per-region attention-to-image fraction (grouped bar per layer).
  3. Combined multi-panel figure.

Usage:
    python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.compare \
        --rl_json /path/to/rl/results.json \
        --base_json /path/to/base/results.json \
        --output_dir /path/to/output
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fire


def compare(
    rl_json: str,
    base_json: str,
    output_dir: str,
):
    with open(rl_json) as f:
        rl_data = json.load(f)
    with open(base_json) as f:
        base_data = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_summary = rl_data["summary"]
    base_summary = base_data["summary"]

    layer_keys = sorted(rl_summary.keys(), key=lambda k: int(k))

    _plot_per_turn_lines(rl_summary, base_summary, layer_keys, out_dir)
    _plot_per_region_bars(rl_summary, base_summary, layer_keys, out_dir)
    _plot_combined(rl_summary, base_summary, layer_keys, out_dir)


def _plot_per_turn_lines(rl_summary, base_summary, layer_keys, out_dir):
    """Line plot: response→image fraction vs turn index, one subplot per layer."""
    n_layers = len(layer_keys)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, lk in zip(axes, layer_keys):
        rl_turns = rl_summary[lk]["per_turn"]
        base_turns = base_summary[lk]["per_turn"]

        # RL
        rl_x = [t["turn_idx"] for t in rl_turns]
        rl_y = [t["mean"] for t in rl_turns]
        rl_err = [t["std"] for t in rl_turns]
        ax.errorbar(rl_x, rl_y, yerr=rl_err, marker='o', capsize=3,
                    label="RL", color="#3B82F6", linewidth=2)

        # Base
        base_x = [t["turn_idx"] for t in base_turns]
        base_y = [t["mean"] for t in base_turns]
        base_err = [t["std"] for t in base_turns]
        ax.errorbar(base_x, base_y, yerr=base_err, marker='s', capsize=3,
                    label="Base", color="#F97316", linewidth=2)

        ax.set_xlabel("Response Turn", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Image Attention Fraction", fontsize=11)
        ax.set_title(f"Layer {lk}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(max(len(rl_x), len(base_x))))

    fig.suptitle("Response → Image Attention Fraction by Turn",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "per_turn_lines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'per_turn_lines.png'}")


def _plot_per_region_bars(rl_summary, base_summary, layer_keys, out_dir):
    """Bar plot: image attention fraction for response regions vs image regions, per layer."""
    # For each layer, extract response_N and image_N regions
    for lk in layer_keys:
        rl_regions = rl_summary[lk]["per_region"]
        base_regions = base_summary[lk]["per_region"]

        # Filter response regions (both models)
        resp_keys_rl = sorted([k for k in rl_regions if k.startswith("response_")],
                              key=lambda k: int(k.split("_")[1]))
        resp_keys_base = sorted([k for k in base_regions if k.startswith("response_")],
                                key=lambda k: int(k.split("_")[1]))
        # Use all turns present in either
        all_resp_indices = sorted(set(
            [int(k.split("_")[1]) for k in resp_keys_rl] +
            [int(k.split("_")[1]) for k in resp_keys_base]
        ))

        fig, ax = plt.subplots(figsize=(max(10, len(all_resp_indices) * 1.5), 5))
        x = np.arange(len(all_resp_indices))
        width = 0.35

        rl_vals = [rl_regions.get(f"response_{idx}", {}).get("mean", 0) for idx in all_resp_indices]
        rl_stds = [rl_regions.get(f"response_{idx}", {}).get("std", 0) for idx in all_resp_indices]
        base_vals = [base_regions.get(f"response_{idx}", {}).get("mean", 0) for idx in all_resp_indices]
        base_stds = [base_regions.get(f"response_{idx}", {}).get("std", 0) for idx in all_resp_indices]

        ax.bar(x - width / 2, rl_vals, width, yerr=rl_stds,
               label="RL", color="#3B82F6", alpha=0.85, capsize=3,
               edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, base_vals, width, yerr=base_stds,
               label="Base", color="#F97316", alpha=0.85, capsize=3,
               edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Response Turn", fontsize=11)
        ax.set_ylabel("Image Attention Fraction", fontsize=11)
        ax.set_title(f"Layer {lk}: Response Region → Image Attention", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Turn {idx}" for idx in all_resp_indices])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"per_region_layer{lk}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / f'per_region_layer{lk}.png'}")


def _plot_combined(rl_summary, base_summary, layer_keys, out_dir):
    """Combined figure: per-turn lines for all layers + global fraction bar."""
    n_layers = len(layer_keys)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: per-turn lines for first 3 layers
    # Bottom row: per-turn lines for remaining layers + global bar
    all_axes = axes.flat
    ax_idx = 0

    for lk in layer_keys:
        if ax_idx >= 5:
            break
        ax = all_axes[ax_idx]

        rl_turns = rl_summary[lk]["per_turn"]
        base_turns = base_summary[lk]["per_turn"]

        rl_x = [t["turn_idx"] for t in rl_turns]
        rl_y = [t["mean"] for t in rl_turns]
        rl_err = [t["std"] for t in rl_turns]
        ax.errorbar(rl_x, rl_y, yerr=rl_err, marker='o', capsize=3,
                    label="RL", color="#3B82F6", linewidth=1.5, markersize=4)

        base_x = [t["turn_idx"] for t in base_turns]
        base_y = [t["mean"] for t in base_turns]
        base_err = [t["std"] for t in base_turns]
        ax.errorbar(base_x, base_y, yerr=base_err, marker='s', capsize=3,
                    label="Base", color="#F97316", linewidth=1.5, markersize=4)

        ax.set_title(f"Layer {lk}", fontsize=11)
        ax.set_xlabel("Turn", fontsize=9)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Img Attn Fraction", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax_idx += 1

    # Last subplot: global mean fraction per layer (bar)
    ax = all_axes[ax_idx]
    x = np.arange(len(layer_keys))
    width = 0.35
    rl_global = [rl_summary[k]["global_mean_fraction"]["mean"] for k in layer_keys]
    rl_global_std = [rl_summary[k]["global_mean_fraction"]["std"] for k in layer_keys]
    base_global = [base_summary[k]["global_mean_fraction"]["mean"] for k in layer_keys]
    base_global_std = [base_summary[k]["global_mean_fraction"]["std"] for k in layer_keys]

    ax.bar(x - width / 2, rl_global, width, yerr=rl_global_std,
           label="RL", color="#3B82F6", alpha=0.85, capsize=3,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, base_global, width, yerr=base_global_std,
           label="Base", color="#F97316", alpha=0.85, capsize=3,
           edgecolor="black", linewidth=0.5)
    ax.set_title("Global Mean Fraction (all positions)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{k}" for k in layer_keys])
    ax.set_ylabel("Img Attn Fraction", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    n_rl = rl_summary[layer_keys[0]]["n_trajectories"]
    n_base = base_summary[layer_keys[0]]["n_trajectories"]
    fig.suptitle(f"Full-Sequence Image Attention: RL (n={n_rl}) vs Base (n={n_base})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'combined.png'}")


if __name__ == "__main__":
    fire.Fire(compare)
