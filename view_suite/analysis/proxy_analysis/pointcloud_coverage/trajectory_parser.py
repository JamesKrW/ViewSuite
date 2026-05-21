"""
Trajectory parsing: reads rollout directories and extracts per-turn camera poses.

Each trajectory is parsed into a TrajectoryInfo holding:
  - init_view / top_down_view c2w matrices (from JSONL, exact 4x4)
  - per-turn camera c2w matrices (from messages.json SE(3) text, converted)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Reuse the project's SE(3)->c2w conversion
from view_suite.scannet.utils.pose_utils import c2w_se3_to_extrinsic


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryInfo:
    """Parsed trajectory with all camera poses needed for visibility analysis."""

    traj_id: str                          # folder name, e.g. "20260204-051746-041ac3b5"
    scene_id: str                         # e.g. "scene0598_00"
    jsonl_idx: int                        # line index in the evaluation JSONL
    sample_id: str                        # e.g. "scene0518_00_sample_016"

    init_view_c2w: np.ndarray             # 4x4 camera-to-world (from JSONL, exact)
    top_down_view_c2w: np.ndarray         # 4x4 camera-to-world (from JSONL, exact)
    target_view_c2w: np.ndarray           # 4x4 camera-to-world (from JSONL, exact)
    turn_c2ws: List[np.ndarray] = field(default_factory=list)  # 4x4 per "Current camera" observation


# ---------------------------------------------------------------------------
# Regex for extracting SE(3) from message text
# ---------------------------------------------------------------------------

# Matches the "Current camera" block specifically:
#   Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
#   [tx=4.0547, ty=3.4760, tz=1.3183, rx=-120.00°, ry=0.00°, rz=150.00°]
#
# This two-line pattern avoids accidentally matching answer-prediction or
# ground-truth SE(3) values that also use the [tx=...] format in the same message.
_CURRENT_CAMERA_PATTERN = re.compile(
    r"Current camera[^\n]*\n"
    r"\[tx=([-\d.]+),\s*ty=([-\d.]+),\s*tz=([-\d.]+),\s*"
    r"rx=([-\d.]+)°?,\s*ry=([-\d.]+)°?,\s*rz=([-\d.]+)°?\]"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    """Extract all text from a message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _parse_current_camera_poses(messages: List[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Extract 'Current camera' SE(3) poses from user messages.

    Includes all observation turns (valid actions, format-error retries, etc.)
    but excludes the answer turn (which starts with "[answer]" and does not
    produce a new observation).

    Returns a list of 4x4 c2w matrices, one per observation turn.
    """
    poses: List[np.ndarray] = []
    answered = False

    for msg in messages:
        role = msg.get("role", "")

        # Once the assistant has output answer(...), discard all subsequent messages.
        if role == "assistant":
            text = _extract_text(msg.get("content", ""))
            if "answer(" in text:
                answered = True
            continue

        if role != "user" or answered:
            continue

        text = _extract_text(msg.get("content", ""))
        match = _CURRENT_CAMERA_PATTERN.search(text)
        if match is None:
            continue

        se3_deg = np.array([float(match.group(i)) for i in range(1, 7)], dtype=np.float64)
        c2w = c2w_se3_to_extrinsic(se3_deg, degrees=True)
        poses.append(c2w)

    return poses


def _resolve_jsonl_path(raw_path: str) -> Path:
    """Resolve a JSONL path, trying common path remappings if it doesn't exist."""
    p = Path(raw_path)
    if p.exists():
        return p
    # Common remapping: /root/projects/data/... → /root/projects/viewsuite/data/...
    alt = Path(str(p).replace("/root/projects/data/", "/root/projects/viewsuite/data/"))
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"JSONL not found at {p} or {alt}"
    )


def _load_jsonl_index(jsonl_path: str | Path) -> Dict[int, Dict[str, Any]]:
    """Load the entire JSONL file into a dict keyed by line index."""
    data: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                data[idx] = json.loads(line)
    return data


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class TrajectoryParser:
    """Parses rollout directories into TrajectoryInfo objects."""

    def __init__(self, jsonl_path: Optional[str | Path] = None):
        """
        Args:
            jsonl_path: Path to the JSONL file used during evaluation.
                        If None, the parser auto-detects the JSONL from each
                        trajectory's env_config.jsonl_path in metrics.json.
        """
        self._explicit_jsonl = Path(jsonl_path) if jsonl_path else None
        # Cache: resolved_path -> {idx: row}
        self._jsonl_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def _get_jsonl_data(self, jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
        """Load and cache a JSONL file by resolved path."""
        key = str(jsonl_path)
        if key not in self._jsonl_cache:
            self._jsonl_cache[key] = _load_jsonl_index(jsonl_path)
        return self._jsonl_cache[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_all(self, rollout_dir: str | Path) -> List[TrajectoryInfo]:
        """
        Parse every trajectory folder under *rollout_dir*.

        Returns:
            Sorted list of TrajectoryInfo (sorted by traj_id).
        """
        rollout_dir = Path(rollout_dir)
        trajectories: List[TrajectoryInfo] = []
        skipped = 0

        # Each sub-directory is one trajectory (skip summary.json etc.)
        for entry in sorted(rollout_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                traj = self._parse_one(entry)
                trajectories.append(traj)
            except Exception as e:
                print(f"[WARN] Skipping {entry.name}: {e}")
                skipped += 1

        print(f"Parsed {len(trajectories)} trajectories ({skipped} skipped)")
        return trajectories

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_one(self, traj_dir: Path) -> TrajectoryInfo:
        """Parse a single trajectory directory."""

        # --- metrics.json: scene_id and jsonl_idx ---
        with open(traj_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
        info0 = metrics["infos"][0]
        scene_id: str = info0["scene_id"]
        jsonl_idx: int = info0["jsonl_idx"]
        sample_id: str = info0.get("sample_id", "")

        # --- Resolve JSONL path (explicit or from env_config) ---
        jsonl_path = self._explicit_jsonl
        if jsonl_path is None:
            env_cfg = metrics.get("env_config", {})
            raw = env_cfg.get("jsonl_path", "")
            if not raw:
                raise ValueError(f"No jsonl_path in env_config and none provided explicitly")
            jsonl_path = _resolve_jsonl_path(raw)

        # --- JSONL entry: exact c2w matrices for init_view and top_down_view ---
        item = self._get_jsonl_data(jsonl_path)[jsonl_idx]
        details = item["image_detail"]
        init_c2w = np.array(details["init_view"]["c2w_extrinsics"], dtype=np.float64)
        topdown_c2w = np.array(details["top_down_view"]["c2w_extrinsics"], dtype=np.float64)
        target_c2w = np.array(details["target_view"]["c2w_extrinsics"], dtype=np.float64)

        # --- messages.json: per-turn camera poses ---
        # Every user message after the initial setup contains a "Current camera"
        # block — including format-error retries (same pose) and the answer turn
        # (same pose, 0 new vertices).  We simply collect them all; duplicate
        # poses naturally produce zero increment in the cumulative union.
        with open(traj_dir / "messages.json", "r") as f:
            messages = json.load(f)
        turn_c2ws = _parse_current_camera_poses(messages)

        return TrajectoryInfo(
            traj_id=traj_dir.name,
            scene_id=scene_id,
            jsonl_idx=jsonl_idx,
            sample_id=sample_id,
            init_view_c2w=init_c2w,
            top_down_view_c2w=topdown_c2w,
            target_view_c2w=target_c2w,
            turn_c2ws=turn_c2ws,
        )
