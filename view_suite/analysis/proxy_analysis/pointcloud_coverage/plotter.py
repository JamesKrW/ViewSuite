"""
Plotting: generates publication-quality visualisations of point-cloud
coverage over turns.

All plots use matplotlib with a clean, minimal style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt


class Plotter:
    """Generates analysis plots from trajectory results and summary statistics."""

    # Consistent styling
    _FIG_SIZE = (10, 6)
    _DPI = 150
    _ALPHA_INDIVIDUAL = 0.08  # transparency for individual trajectory lines

    @staticmethod
    def plot_avg_cumulative(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_cumulative.png",
    ) -> None:
        """Plot mean cumulative visible-vertex count (+/- 1 std) vs. turn."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["cumulative"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "o-", color="#2563EB", linewidth=2, markersize=5, label="Mean")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#2563EB", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Cumulative visible vertices", fontsize=12)
        ax.set_title("Average Cumulative Point Cloud Coverage by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_increment(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_increment.png",
    ) -> None:
        """Plot mean per-turn increment (+/- 1 std) vs. turn."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["increment"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.bar(turns, means, color="#10B981", alpha=0.7, label="Mean increment")
        ax.errorbar(turns, means, yerr=stds, fmt="none", ecolor="gray", capsize=3, label="±1 std")
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("New vertices this turn", fontsize=12)
        ax.set_title("Average Per-Turn Increment in Visible Vertices", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_coverage_ratio(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_coverage_ratio.png",
    ) -> None:
        """Plot mean coverage ratio (fraction of total mesh vertices) vs. turn."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["coverage_ratio"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "s-", color="#F59E0B", linewidth=2, markersize=5, label="Mean ratio")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#F59E0B", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Coverage ratio (visible / total)", fontsize=12)
        ax.set_title("Average Coverage Ratio by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_all_trajectories(
        trajectory_results: List[Dict[str, Any]],
        output_dir: str | Path,
        filename: str = "all_trajectories.png",
    ) -> None:
        """Overlay all individual cumulative curves (semi-transparent)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)

        # Adapt alpha to trajectory count: visible for small N, dense overlay for large N
        n = len(trajectory_results)
        alpha = max(0.05, min(0.8, 5.0 / n)) if n > 0 else 0.5

        for r in trajectory_results:
            turns = list(range(r["num_turns"]))
            ax.plot(
                turns, r["cumulative_counts"],
                alpha=alpha, color="#6366F1", linewidth=0.8,
            )

        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Cumulative visible vertices", fontsize=12)
        ax.set_title(f"All Trajectories — Cumulative Point Cloud Coverage ({len(trajectory_results)} trajs)", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Determine max turn for x-axis
        max_turns = max((r["num_turns"] for r in trajectory_results), default=0)
        if max_turns > 0:
            _set_integer_xticks(ax, list(range(max_turns)))

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_coverage_ratio_inc(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_coverage_ratio_inc.png",
    ) -> None:
        """Plot mean per-turn coverage ratio increment vs. turn."""
        if "coverage_ratio_inc" not in summary:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["coverage_ratio_inc"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "o-", color="#F59E0B", linewidth=2, markersize=5, label="Mean")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#F59E0B", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Coverage ratio increment per turn", fontsize=12)
        ax.set_title("Average Coverage Ratio Increment by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    # ------------------------------------------------------------------
    # Target intersection plots
    # ------------------------------------------------------------------

    @staticmethod
    def plot_avg_target_cumulative(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_target_cumulative.png",
    ) -> None:
        """Plot mean cumulative target intersection count vs. turn."""
        if "target_intersection" not in summary:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["target_intersection"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "o-", color="#8B5CF6", linewidth=2, markersize=5, label="Mean")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#8B5CF6", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Target intersection vertices (|cum ∩ target|)", fontsize=12)
        ax.set_title("Average Cumulative Target Intersection by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_target_increment(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_target_increment.png",
    ) -> None:
        """Plot mean per-turn target intersection increment vs. turn."""
        if "target_intersection_inc" not in summary:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["target_intersection_inc"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.bar(turns, means, color="#8B5CF6", alpha=0.7, label="Mean increment")
        ax.errorbar(turns, means, yerr=stds, fmt="none", ecolor="gray", capsize=3, label="±1 std")
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("New target-intersection vertices this turn", fontsize=12)
        ax.set_title("Average Per-Turn Target Intersection Increment", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_target_ratio(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_target_ratio.png",
    ) -> None:
        """Plot mean target-intersection ratio (|cum ∩ target| / |target|) vs. turn."""
        if "target_intersection_ratio" not in summary:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["target_intersection_ratio"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "D-", color="#8B5CF6", linewidth=2, markersize=5, label="Mean ratio")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#8B5CF6", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Target intersection ratio (|cum ∩ target| / |target|)", fontsize=12)
        ax.set_title("Average Target View Intersection Ratio by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_avg_target_ratio_inc(
        summary: Dict[str, Any],
        output_dir: str | Path,
        filename: str = "avg_target_ratio_inc.png",
    ) -> None:
        """Plot mean per-turn target intersection ratio increment vs. turn."""
        if "target_intersection_ratio_inc" not in summary:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        turns, means, stds = _extract_series(summary["target_intersection_ratio_inc"])

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)
        ax.plot(turns, means, "o-", color="#8B5CF6", linewidth=2, markersize=5, label="Mean")
        ax.fill_between(
            turns,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2, color="#8B5CF6", label="±1 std",
        )
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Target intersection ratio increment per turn", fontsize=12)
        ax.set_title("Average Target Intersection Ratio Increment by Turn", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        _set_integer_xticks(ax, turns)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_all_trajectories_target(
        trajectory_results: List[Dict[str, Any]],
        output_dir: str | Path,
        filename: str = "all_trajectories_target.png",
    ) -> None:
        """Overlay all individual target intersection ratio curves."""
        if not trajectory_results or "target_intersection_ratios" not in trajectory_results[0]:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=Plotter._FIG_SIZE)

        n = len(trajectory_results)
        alpha = max(0.05, min(0.8, 5.0 / n)) if n > 0 else 0.5

        for r in trajectory_results:
            turns = list(range(r["num_turns"]))
            ax.plot(
                turns, r["target_intersection_ratios"],
                alpha=alpha, color="#8B5CF6", linewidth=0.8,
            )

        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Target intersection ratio (|cum ∩ target| / |target|)", fontsize=12)
        ax.set_title(f"All Trajectories — Target Intersection Ratio ({n} trajs)", fontsize=14)
        ax.grid(True, alpha=0.3)

        max_turns = max((r["num_turns"] for r in trajectory_results), default=0)
        if max_turns > 0:
            _set_integer_xticks(ax, list(range(max_turns)))

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=Plotter._DPI)
        plt.close(fig)
        print(f"Saved {output_dir / filename}")

    @staticmethod
    def plot_all(
        trajectory_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        """Generate all standard plots at once."""
        # Global coverage
        Plotter.plot_avg_cumulative(summary, output_dir)
        Plotter.plot_avg_increment(summary, output_dir)
        Plotter.plot_avg_coverage_ratio(summary, output_dir)
        Plotter.plot_avg_coverage_ratio_inc(summary, output_dir)
        Plotter.plot_all_trajectories(trajectory_results, output_dir)
        # Target intersection
        Plotter.plot_avg_target_cumulative(summary, output_dir)
        Plotter.plot_avg_target_increment(summary, output_dir)
        Plotter.plot_avg_target_ratio(summary, output_dir)
        Plotter.plot_avg_target_ratio_inc(summary, output_dir)
        Plotter.plot_all_trajectories_target(trajectory_results, output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_series(stats_by_turn: Dict[str, Dict[str, Any]]):
    """
    Extract sorted turn indices, means, and stds from a summary sub-dict.

    Returns:
        (turns, means, stds) — all lists of the same length.
    """
    turns = sorted(int(k) for k in stats_by_turn.keys())
    means = [stats_by_turn[str(t)]["mean"] for t in turns]
    stds = [stats_by_turn[str(t)]["std"] for t in turns]
    return turns, means, stds


def _set_integer_xticks(ax, turns):
    """Set x-axis ticks to integer turn indices."""
    if turns:
        ax.set_xticks(turns)
        ax.set_xticklabels([str(t) for t in turns])
