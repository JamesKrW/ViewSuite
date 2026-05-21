"""
Compare two models' image attention results and generate plots.

Usage:
    python -m view_suite.analysis.scannet_attn_image.compare \
        --rl_json /path/to/rl/attn_image_analysis/results.json \
        --base_json /path/to/base/attn_image_analysis/results.json \
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

    n_rl = rl_summary[layer_keys[0]]["n_trajectories"]
    n_base = base_summary[layer_keys[0]]["n_trajectories"]

    # --- Individual metric plots ---
    metrics = [
        ("attn_mean", "Mean Attention to Image Tokens", True),
        ("attn_max", "Max Attention to Image Tokens", True),
        ("attn_min", "Min Attention to Image Tokens", True),
        ("img_attn_fraction", "Image Attention Fraction (sum / total)", False),
    ]

    for metric_key, metric_title, use_log in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        _draw_bars(ax, rl_summary, base_summary, layer_keys,
                   metric_key, metric_title, use_log)
        ax.set_title(f"{metric_title}\nRL (n={n_rl}) vs Base (n={n_base})", fontsize=13)
        fig.tight_layout()
        fname = f"compare_{metric_key}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / fname}")

    # --- Combined 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (metric_key, metric_title, use_log) in zip(axes.flat, metrics):
        _draw_bars(ax, rl_summary, base_summary, layer_keys,
                   metric_key, metric_title, use_log)
        ax.set_title(metric_title, fontsize=11)

    fig.suptitle(f"Image Attention: RL (n={n_rl}) vs Base (n={n_base})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'compare_all.png'}")

    # --- Print img_token_counts for sanity check ---
    print(f"\nImage token counts (sanity check):")
    for k in layer_keys:
        rl_counts = rl_summary[k].get("img_token_counts", [])
        base_counts = base_summary[k].get("img_token_counts", [])
        print(f"  Layer {k}: RL={rl_counts}, Base={base_counts}")


def _draw_bars(ax, rl_summary, base_summary, layer_keys,
               metric_key, metric_title, use_log):
    x = np.arange(len(layer_keys))
    width = 0.35

    rl_means = [rl_summary[k][metric_key]["mean"] for k in layer_keys]
    rl_stds = [rl_summary[k][metric_key]["std"] for k in layer_keys]
    base_means = [base_summary[k][metric_key]["mean"] for k in layer_keys]
    base_stds = [base_summary[k][metric_key]["std"] for k in layer_keys]

    ax.bar(x - width / 2, rl_means, width, yerr=rl_stds,
           label="RL (iter5_rl_960)", color="#3B82F6",
           alpha=0.85, capsize=4, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, base_means, width, yerr=base_stds,
           label="Base (Qwen2.5-VL-7B)", color="#F97316",
           alpha=0.85, capsize=4, edgecolor="black", linewidth=0.5)

    if use_log:
        ax.set_yscale("log")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(metric_title + (" (log)" if use_log else ""), fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {k}" for k in layer_keys])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)


if __name__ == "__main__":
    fire.Fire(compare)
