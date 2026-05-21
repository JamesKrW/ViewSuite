"""
Compare coverage curves across multiple rollouts on the same plot.

Usage:
    python -m view_suite.analysis.scannet_point_by_turn.compare \
        /root/projects/viewsuite/data/rollouts/sft_gen_gemini_3_pro/tag_0_filter/coverage_analysis \
        /root/projects/viewsuite/data/rollouts/tag_ae_example_loose_no_example/coverage_analysis \
        --output_dir ./compare_output \
        --labels "Gemini 3 Pro SFT (2686 trajs)" "Qwen VL 7B (530 trajs)"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fire

# Consistent styling
_FIG_SIZE = (10, 6)
_DPI = 150
_COLORS = [
    "#2563EB",  # blue
    "#EF4444",  # red
    "#10B981",  # green
    "#F59E0B",  # amber
    "#8B5CF6",  # purple
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
    "#6366F1",  # indigo
    "#14B8A6",  # teal
    "#E11D48",  # rose
    "#84CC16",  # lime
    "#A855F7",  # violet
    "#0EA5E9",  # sky
    "#D946EF",  # fuchsia
    "#22C55E",  # emerald
]


def _load_summary(coverage_dir: str | Path) -> Dict[str, Any]:
    p = Path(coverage_dir) / "summary.json"
    with open(p) as f:
        return json.load(f)


def _extract_series(stats_by_turn: Dict[str, Dict[str, Any]]):
    turns = sorted(int(k) for k in stats_by_turn.keys())
    means = [stats_by_turn[str(t)]["mean"] for t in turns]
    stds = [stats_by_turn[str(t)]["std"] for t in turns]
    counts = [stats_by_turn[str(t)]["count"] for t in turns]
    return turns, means, stds, counts


def _auto_label(coverage_dir: str | Path) -> str:
    """Derive a short label from the rollout directory path."""
    p = Path(coverage_dir)
    # Go up from coverage_analysis to the rollout dir
    if p.name == "coverage_analysis":
        p = p.parent
    # Use the last 1-2 path components
    parts = p.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1] if parts else str(coverage_dir)


def _set_integer_xticks(ax, turns):
    if turns:
        ax.set_xticks(turns)
        ax.set_xticklabels([str(t) for t in turns])


def compare(
    *coverage_dirs: str,
    output_dir: str = "./compare_output",
    labels: Optional[str] = None,
) -> None:
    """
    Overlay mean coverage curves from multiple rollouts.

    Args:
        coverage_dirs: One or more paths to coverage_analysis/ directories
                       (each must contain summary.json).
        output_dir:    Where to save comparison plots.
        labels:        Comma-separated display names for each rollout.
                       E.g. "Gemini 3 Pro SFT,Qwen VL 7B".
                       If not provided, auto-derived from directory paths.
    """
    if not coverage_dirs:
        print("Usage: compare <dir1> <dir2> [--output_dir ...] [--labels ...]")
        sys.exit(1)

    dirs = list(coverage_dirs)
    if labels is None:
        label_list = [_auto_label(d) for d in dirs]
    else:
        label_list = [l.strip() for l in labels.split(",")]
        if len(label_list) != len(dirs):
            print(f"Error: {len(label_list)} labels provided but {len(dirs)} directories given.")
            sys.exit(1)

    summaries = []
    for d in dirs:
        s = _load_summary(d)
        summaries.append(s)
        print(f"Loaded: {d} ({s['num_trajectories']} trajectories, max {s['max_turns']} turns)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_cumulative(summaries, label_list, out)
    _plot_increment(summaries, label_list, out)
    _plot_coverage_ratio(summaries, label_list, out)

    # Target intersection plots (only if data available)
    if any("target_intersection_ratio" in s for s in summaries):
        _plot_target_cumulative(summaries, label_list, out)
        _plot_target_increment(summaries, label_list, out)
        _plot_target_intersection_ratio(summaries, label_list, out)

    print(f"\nAll comparison plots saved to {out}")


def _plot_cumulative(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    for i, (s, label) in enumerate(zip(summaries, labels)):
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["cumulative"])
        # Filter out turns with very few trajectories (< 1% of max)
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        n = s["num_trajectories"]
        ax.plot(turns, means, "o-", color=color, linewidth=2, markersize=4,
                label=f"{label} (n={n})")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Cumulative visible vertices", fontsize=12)
    ax.set_title("Cumulative Point Cloud Coverage — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_turns = set()
    for s in summaries:
        all_turns.update(int(k) for k in s["cumulative"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_cumulative.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_increment(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    n_runs = len(summaries)
    bar_width = 0.8 / n_runs

    for i, (s, label) in enumerate(zip(summaries, labels)):
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["increment"])
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        offset = (i - (n_runs - 1) / 2) * bar_width
        positions = np.array(turns) + offset
        n = s["num_trajectories"]
        ax.bar(positions, means, width=bar_width * 0.9, color=color, alpha=0.7,
               label=f"{label} (n={n})")
        ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor="gray",
                    capsize=2, linewidth=0.8)

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("New vertices this turn", fontsize=12)
    ax.set_title("Per-Turn Increment — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    all_turns = set()
    for s in summaries:
        all_turns.update(int(k) for k in s["increment"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_increment.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_coverage_ratio(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    for i, (s, label) in enumerate(zip(summaries, labels)):
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["coverage_ratio"])
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        n = s["num_trajectories"]
        ax.plot(turns, means, "s-", color=color, linewidth=2, markersize=4,
                label=f"{label} (n={n})")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Coverage ratio (visible / total)", fontsize=12)
    ax.set_title("Coverage Ratio — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_turns = set()
    for s in summaries:
        all_turns.update(int(k) for k in s["coverage_ratio"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_coverage_ratio.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_target_cumulative(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    for i, (s, label) in enumerate(zip(summaries, labels)):
        if "target_intersection" not in s:
            continue
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["target_intersection"])
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        n = s["num_trajectories"]
        ax.plot(turns, means, "o-", color=color, linewidth=2, markersize=4,
                label=f"{label} (n={n})")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Target intersection vertices (|cum ∩ target|)", fontsize=12)
    ax.set_title("Cumulative Target Intersection — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_turns = set()
    for s in summaries:
        if "target_intersection" in s:
            all_turns.update(int(k) for k in s["target_intersection"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_target_cumulative.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_target_increment(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    n_runs = len(summaries)
    bar_width = 0.8 / n_runs

    for i, (s, label) in enumerate(zip(summaries, labels)):
        if "target_intersection_inc" not in s:
            continue
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["target_intersection_inc"])
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        offset = (i - (n_runs - 1) / 2) * bar_width
        positions = np.array(turns) + offset
        n = s["num_trajectories"]
        ax.bar(positions, means, width=bar_width * 0.9, color=color, alpha=0.7,
               label=f"{label} (n={n})")
        ax.errorbar(positions, means, yerr=stds, fmt="none", ecolor="gray",
                    capsize=2, linewidth=0.8)

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("New target-intersection vertices this turn", fontsize=12)
    ax.set_title("Per-Turn Target Intersection Increment — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    all_turns = set()
    for s in summaries:
        if "target_intersection_inc" in s:
            all_turns.update(int(k) for k in s["target_intersection_inc"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_target_increment.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_target_intersection_ratio(
    summaries: List[Dict], labels: List[str], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    for i, (s, label) in enumerate(zip(summaries, labels)):
        if "target_intersection_ratio" not in s:
            continue
        color = _COLORS[i % len(_COLORS)]
        turns, means, stds, counts = _extract_series(s["target_intersection_ratio"])
        max_count = max(counts)
        valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
        turns = [turns[j] for j in valid]
        means = [means[j] for j in valid]
        stds = [stds[j] for j in valid]

        n = s["num_trajectories"]
        ax.plot(turns, means, "D-", color=color, linewidth=2, markersize=4,
                label=f"{label} (n={n})")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Turn", fontsize=12)
    ax.set_ylabel("Target intersection ratio (|cum ∩ target| / |target|)", fontsize=12)
    ax.set_title("Target View Intersection Ratio — Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_turns = set()
    for s in summaries:
        if "target_intersection_ratio" in s:
            all_turns.update(int(k) for k in s["target_intersection_ratio"].keys())
    _set_integer_xticks(ax, sorted(all_turns))

    fig.tight_layout()
    path = output_dir / "compare_target_intersection_ratio.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fire.Fire(compare)
