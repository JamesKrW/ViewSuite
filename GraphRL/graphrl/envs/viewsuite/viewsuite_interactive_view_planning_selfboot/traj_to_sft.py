"""
Self-bootstrapping baseline TrajToSFT phase.

Different from the graph-based envs: this one uses ``SelfBootFilterBuilder``
to filter high-reward trajectories from VAGEN rollouts (rather than build a
graph), then converts them directly to LLaMA-Factory ShareGPT records.

Pipeline.yaml::

    traj_to_sft:
      module: graphrl.envs.viewsuite.viewsuite_interactive_view_planning_selfboot.SelfBootTrajToSFT
      reward_threshold: 0.5
      # filter_builder: extra config passed to SelfBootFilterBuilder
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from graphrl import TrajToSFTModule
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning_selfboot.selfboot_filter_builder import (
    SelfBootFilterBuilder,
)

logger = logging.getLogger(__name__)

_SHAREGPT_FMT = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}

DATASET_NAME = "selfboot_multi_turn"


class SelfBootTrajToSFT(TrajToSFTModule):
    """Filter high-reward VAGEN trajectories → ShareGPT SFT data."""

    name = "TrajToSFT(selfboot)"

    def run(self) -> None:
        graph_dir = self._build_or_load_filtered()
        records = self._build_records(graph_dir)
        if not records:
            raise RuntimeError("No valid SFT records produced from filtered trajectories.")

        # Use the base helper for dataset_info.json — but record images already
        # copied into the dataset subdir, so we just write the records JSON
        # under the per-dataset name directly.
        self.write_dataset_info({DATASET_NAME: (records, _SHAREGPT_FMT)})

    # ── stage 1: filter rollouts → graph_dir/filtered_trajs.json ──────────

    def _build_or_load_filtered(self) -> Path:
        graph_dir = self.paths.base_dir / "rl" / "graph"
        filtered_path = graph_dir / "filtered_trajs.json"

        rollouts = sorted(self.paths.rollout_data.glob("*.jsonl"))
        if rollouts:
            graph_dir.mkdir(parents=True, exist_ok=True)
            builder = SelfBootFilterBuilder(self.config)
            self._log(f"Filtering {len(rollouts)} rollout file(s)")
            builder.convert_files(rollouts, self.paths.rollout_data, graph_dir)
        elif filtered_path.exists():
            self._log(f"No new rollouts; using cached {filtered_path}")
        else:
            raise RuntimeError(
                f"No rollouts at {self.paths.rollout_data} and no cached "
                f"filtered_trajs.json at {filtered_path}."
            )
        return graph_dir

    # ── stage 2: convert filtered → ShareGPT records ─────────────────────

    def _build_records(self, graph_dir: Path) -> List[Dict[str, Any]]:
        filtered_path = graph_dir / "filtered_trajs.json"
        with filtered_path.open("r") as f:
            filtered = json.load(f)
        if not filtered:
            raise RuntimeError(
                "filtered_trajs.json is empty — no high-reward trajectories found."
            )
        self._log(f"Loaded {len(filtered)} filtered trajectories from {filtered_path}")

        ds_dir = self.paths.sft_data / DATASET_NAME
        ds_dir.mkdir(parents=True, exist_ok=True)

        records: List[Dict[str, Any]] = []
        for traj in filtered:
            messages = traj["messages"]
            image_paths = traj.get("image_paths", []) or []
            if len(messages) < 3:
                continue

            copied: List[str] = []
            for rel_path in image_paths:
                if not rel_path:
                    continue
                src = graph_dir / rel_path
                if not src.exists():
                    continue
                dst = ds_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied.append(f"{DATASET_NAME}/{src.name}")

            records.append({"messages": messages, "images": copied})

        self._log(f"Built {len(records)} SFT records → {self.paths.sft_data}")
        return records
