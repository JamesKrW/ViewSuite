"""
Result aggregation: computes per-turn statistics (mean, std, min, max, count)
from a list of per-trajectory results.

Handles variable-length trajectories: each turn slot only includes
trajectories that actually have data for that turn.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class ResultAggregator:
    """Aggregates per-trajectory results into summary statistics."""

    @staticmethod
    def aggregate(trajectory_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute per-turn statistics across all trajectories.

        Args:
            trajectory_results: List of dicts from TrajectoryResult.to_dict().

        Returns:
            Summary dict with keys:
              - num_trajectories
              - max_turns
              - cumulative:     {turn_idx: {mean, std, min, max, count}}
              - increment:      {turn_idx: {mean, std, min, max, count}}
              - coverage_ratio: {turn_idx: {mean, std, min, max, count}}
        """
        if not trajectory_results:
            return {"num_trajectories": 0, "max_turns": 0,
                    "cumulative": {}, "increment": {}, "coverage_ratio": {},
                    "target_intersection": {}, "target_intersection_inc": {},
                    "target_intersection_ratio": {}}

        max_turns = max(r["num_turns"] for r in trajectory_results)

        cumulative_stats = {}
        increment_stats = {}
        coverage_stats = {}
        coverage_ratio_inc_stats = {}
        target_int_stats = {}
        target_int_inc_stats = {}
        target_int_ratio_stats = {}
        target_int_ratio_inc_stats = {}

        for turn_idx in range(max_turns):
            # Collect values from trajectories that have this turn
            cum_vals = []
            inc_vals = []
            cov_vals = []
            cov_inc_vals = []
            ti_vals = []
            ti_inc_vals = []
            ti_ratio_vals = []
            ti_ratio_inc_vals = []

            for r in trajectory_results:
                if turn_idx < r["num_turns"]:
                    cum_vals.append(r["cumulative_counts"][turn_idx])
                    inc_vals.append(r["increments"][turn_idx])
                    cov_vals.append(r["coverage_ratios"][turn_idx])
                    # Coverage ratio increment (delta from previous turn)
                    if turn_idx == 0:
                        cov_inc_vals.append(r["coverage_ratios"][0])
                    else:
                        cov_inc_vals.append(r["coverage_ratios"][turn_idx] - r["coverage_ratios"][turn_idx - 1])
                    if "target_intersections" in r:
                        ti_vals.append(r["target_intersections"][turn_idx])
                        ti_inc_vals.append(r["target_intersection_incs"][turn_idx])
                        ti_ratio_vals.append(r["target_intersection_ratios"][turn_idx])
                        if turn_idx == 0:
                            ti_ratio_inc_vals.append(r["target_intersection_ratios"][0])
                        else:
                            ti_ratio_inc_vals.append(
                                r["target_intersection_ratios"][turn_idx] - r["target_intersection_ratios"][turn_idx - 1])

            cumulative_stats[str(turn_idx)] = _compute_stats(cum_vals)
            increment_stats[str(turn_idx)] = _compute_stats(inc_vals)
            coverage_stats[str(turn_idx)] = _compute_stats(cov_vals)
            coverage_ratio_inc_stats[str(turn_idx)] = _compute_stats(cov_inc_vals)
            target_int_stats[str(turn_idx)] = _compute_stats(ti_vals)
            target_int_inc_stats[str(turn_idx)] = _compute_stats(ti_inc_vals)
            target_int_ratio_stats[str(turn_idx)] = _compute_stats(ti_ratio_vals)
            target_int_ratio_inc_stats[str(turn_idx)] = _compute_stats(ti_ratio_inc_vals)

        return {
            "num_trajectories": len(trajectory_results),
            "max_turns": max_turns,
            "cumulative": cumulative_stats,
            "increment": increment_stats,
            "coverage_ratio": coverage_stats,
            "coverage_ratio_inc": coverage_ratio_inc_stats,
            "target_intersection": target_int_stats,
            "target_intersection_inc": target_int_inc_stats,
            "target_intersection_ratio": target_int_ratio_stats,
            "target_intersection_ratio_inc": target_int_ratio_inc_stats,
        }

    @staticmethod
    def save_json(
        trajectory_results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_dir: str | Path,
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Save results.json and summary.json to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-trajectory results
        results_payload = {
            "config": config or {},
            "trajectories": trajectory_results,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"Saved {output_dir / 'results.json'}")

        # Summary statistics
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved {output_dir / 'summary.json'}")

    @staticmethod
    def save_csv(
        trajectory_results: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> None:
        """Save a flat CSV with one row per trajectory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        max_turns = max((r["num_turns"] for r in trajectory_results), default=0)

        # Build header
        has_target = any("target_intersections" in r for r in trajectory_results)
        header = ["traj_id", "scene_id", "total_vertices", "num_turns"]
        if has_target:
            header.append("target_vertices")
        for i in range(max_turns):
            header.append(f"cum_turn_{i}")
        for i in range(max_turns):
            header.append(f"inc_turn_{i}")
        for i in range(max_turns):
            header.append(f"cov_turn_{i}")
        if has_target:
            for i in range(max_turns):
                header.append(f"target_int_turn_{i}")
            for i in range(max_turns):
                header.append(f"target_int_ratio_turn_{i}")

        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for r in trajectory_results:
                row = [r["traj_id"], r["scene_id"], r["total_vertices"], r["num_turns"]]
                if has_target:
                    row.append(r.get("target_vertices", ""))
                # Pad shorter trajectories with empty cells
                for i in range(max_turns):
                    row.append(r["cumulative_counts"][i] if i < r["num_turns"] else "")
                for i in range(max_turns):
                    row.append(r["increments"][i] if i < r["num_turns"] else "")
                for i in range(max_turns):
                    row.append(r["coverage_ratios"][i] if i < r["num_turns"] else "")
                if has_target:
                    for i in range(max_turns):
                        row.append(r["target_intersections"][i] if i < r["num_turns"] else "")
                    for i in range(max_turns):
                        row.append(r["target_intersection_ratios"][i] if i < r["num_turns"] else "")
                writer.writerow(row)

        print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stats(values: List[float]) -> Dict[str, Any]:
    """Compute mean/std/min/max/count for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "count": len(values),
    }
