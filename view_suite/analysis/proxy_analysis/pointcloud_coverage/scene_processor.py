"""
Scene-level processing: computes cumulative visible-vertex counts for all
trajectories that share a single ScanNet scene.

Designed to run inside a worker process so that Open3D resources are
isolated per-process (as required by Open3D's global state).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np

from view_suite.analysis.proxy_analysis.pointcloud_coverage.visibility import VisibilityComputer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Output for one trajectory: per-turn cumulative and incremental counts."""

    traj_id: str
    scene_id: str
    jsonl_idx: int                    # line index in the evaluation JSONL
    sample_id: str                    # e.g. "scene0518_00_sample_016"
    total_vertices: int               # mesh vertex count (denominator for coverage)
    cumulative_counts: List[int] = field(default_factory=list)  # |union_0|, |union_1|, ...
    increments: List[int] = field(default_factory=list)         # delta per turn
    # Target-view intersection: |cumulative ∩ target_visible| per turn
    target_vertices: int = 0                                        # |target_visible|
    target_intersections: List[int] = field(default_factory=list)   # per-turn cumulative intersection count
    target_intersection_incs: List[int] = field(default_factory=list)  # per-turn intersection increment

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        coverage = [c / self.total_vertices if self.total_vertices > 0 else 0.0
                    for c in self.cumulative_counts]
        target_ratios = [c / self.target_vertices if self.target_vertices > 0 else 0.0
                         for c in self.target_intersections]
        return {
            "traj_id": self.traj_id,
            "scene_id": self.scene_id,
            "jsonl_idx": self.jsonl_idx,
            "sample_id": self.sample_id,
            "total_vertices": self.total_vertices,
            "num_turns": len(self.cumulative_counts),
            "cumulative_counts": self.cumulative_counts,
            "increments": self.increments,
            "coverage_ratios": [round(r, 6) for r in coverage],
            "target_vertices": self.target_vertices,
            "target_intersections": self.target_intersections,
            "target_intersection_incs": self.target_intersection_incs,
            "target_intersection_ratios": [round(r, 6) for r in target_ratios],
        }


# ---------------------------------------------------------------------------
# Worker function (top-level, picklable for multiprocessing)
# ---------------------------------------------------------------------------

def process_scene(
    scene_id: str,
    mesh_path: str,
    trajectories_data: List[Dict[str, Any]],
    K_3x3: np.ndarray,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    """
    Process all trajectories for one scene.

    This function is the multiprocessing entry point.  It receives plain
    dicts / arrays (not dataclass instances) to avoid pickling issues with
    complex objects across processes.

    Args:
        scene_id:           e.g. "scene0598_00"
        mesh_path:          Absolute path to the scene's .ply mesh.
        trajectories_data:  List of dicts, each with keys:
                              traj_id, init_view_c2w, top_down_view_c2w, turn_c2ws
                            where c2w values are 4x4 np.ndarray (or nested lists).
        K_3x3:              3x3 camera intrinsic matrix.
        width:              Image width for raycasting.
        height:             Image height for raycasting.

    Returns:
        List of result dicts (one per trajectory), each from
        TrajectoryResult.to_dict().
    """
    n_trajs = len(trajectories_data)
    _log(f"[{scene_id}] Loading mesh ({n_trajs} trajectories) ...")

    # Instantiate the VisibilityComputer once for this scene.
    t_mesh = time.time()
    vc = VisibilityComputer(mesh_path, K_3x3, width=width, height=height)
    _log(f"[{scene_id}] Mesh loaded: {vc.total_vertices:,} vertices ({time.time()-t_mesh:.1f}s)")

    results: List[Dict[str, Any]] = []

    for traj_idx, tdata in enumerate(trajectories_data):
        t_traj = time.time()
        traj_id: str = tdata["traj_id"]
        jsonl_idx: int = tdata["jsonl_idx"]
        sample_id: str = tdata.get("sample_id", "")
        init_c2w = np.asarray(tdata["init_view_c2w"], dtype=np.float64)
        topdown_c2w = np.asarray(tdata["top_down_view_c2w"], dtype=np.float64)
        target_c2w = np.asarray(tdata["target_view_c2w"], dtype=np.float64)
        turn_c2ws = [np.asarray(c, dtype=np.float64) for c in tdata["turn_c2ws"]]

        # Compute target view visible vertices (once per trajectory)
        target_visible = vc.get_visible_vertex_indices(target_c2w)

        cumulative: Set[int] = set()
        cumulative_counts: List[int] = []
        increments: List[int] = []
        target_intersections: List[int] = []
        target_intersection_incs: List[int] = []

        # --- Turn 0: init_view ∪ top_down_view ---
        vis_init = vc.get_visible_vertex_indices(init_c2w)
        vis_topdown = vc.get_visible_vertex_indices(topdown_c2w)
        cumulative = vis_init | vis_topdown
        cumulative_counts.append(len(cumulative))
        increments.append(len(cumulative))  # first increment == first count
        ti_count = len(cumulative & target_visible)
        target_intersections.append(ti_count)
        target_intersection_incs.append(ti_count)

        # --- Subsequent turns ---
        for c2w in turn_c2ws:
            prev_count = len(cumulative)
            prev_ti = target_intersections[-1]
            vis_turn = vc.get_visible_vertex_indices(c2w)
            cumulative |= vis_turn
            cumulative_counts.append(len(cumulative))
            increments.append(len(cumulative) - prev_count)
            ti_count = len(cumulative & target_visible)
            target_intersections.append(ti_count)
            target_intersection_incs.append(ti_count - prev_ti)

        result = TrajectoryResult(
            traj_id=traj_id,
            scene_id=scene_id,
            jsonl_idx=jsonl_idx,
            sample_id=sample_id,
            total_vertices=vc.total_vertices,
            cumulative_counts=cumulative_counts,
            increments=increments,
            target_vertices=len(target_visible),
            target_intersections=target_intersections,
            target_intersection_incs=target_intersection_incs,
        )
        results.append(result.to_dict())

        elapsed = time.time() - t_traj
        final_cov = cumulative_counts[-1] / vc.total_vertices if vc.total_vertices > 0 else 0
        final_ti = target_intersections[-1] / len(target_visible) if target_visible else 0
        _log(f"[{scene_id}] traj {traj_idx+1}/{n_trajs} "
             f"({traj_id[:16]}...) {len(cumulative_counts)} turns, "
             f"coverage={final_cov:.1%}, target_hit={final_ti:.1%}, {elapsed:.1f}s")

    _log(f"[{scene_id}] Done — {n_trajs} trajectories")
    return results


def _log(msg: str) -> None:
    """Flush-safe print for worker processes."""
    print(msg, flush=True)
