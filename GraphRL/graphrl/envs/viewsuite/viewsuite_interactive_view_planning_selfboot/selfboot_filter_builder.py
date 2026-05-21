"""
Self-bootstrapping trajectory filter.

Replaces the graph builder in the standard pipeline.  Instead of building
a graph from rollout trajectories, it:

  1. Reads raw JSONL rollout files produced by VAGEN.
  2. Filters trajectories whose total reward exceeds a threshold (default 0.5).
  3. Saves the filtered conversations (parsed from ChatML) and their images
     into ``graph_dir/`` so the downstream SFT generator can consume them.

Registered as ``viewsuite_interactive_view_planning_selfboot`` in graph_builder_registry.

The output in graph_dir/:
    filtered_trajs.json   — list of {messages, image_paths, reward}
    images/               — copied observation images
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# SelfBootFilterBuilder is a plain utility class — it does NOT inherit from
# VagenGraphBuilder because it doesn't actually build a graph, it just filters
# high-reward trajectories into ``filtered_trajs.json``.

logger = logging.getLogger(__name__)

_CHATML_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_IMAGE_TAG_RE = re.compile(r"<image>")


def _parse_chatml(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Parse ChatML-encoded VAGEN JSONL line → list of message dicts."""
    full = (data.get("input", "") + data.get("output", "")).replace("<|endoftext|>", "")
    return [
        {"role": role, "content": content.strip()}
        for role, content in _CHATML_RE.findall(full)
    ]


def _get_reward(data: Dict[str, Any]) -> float:
    """Extract scalar reward from a VAGEN JSONL line.

    VAGEN stores the trajectory reward in the ``score`` field (a plain float).
    Falls back to summing ``reward_scores`` dict values if ``score`` is absent.
    """
    # Primary: "score" field (plain float, used by VAGEN active exploration)
    if "score" in data:
        return float(data["score"])
    # Fallback: "reward_scores" dict (used by some other envs)
    rs = data.get("reward_scores", {})
    if isinstance(rs, (int, float)):
        return float(rs)
    if isinstance(rs, dict):
        total = 0.0
        for v in rs.values():
            if isinstance(v, (int, float)):
                total += v
            elif isinstance(v, list):
                total += sum(x for x in v if isinstance(x, (int, float)))
        return total
    return 0.0


class SelfBootFilterBuilder:
    """Filter high-reward trajectories and save them for SFT."""

    def __init__(self, config):
        self.config = config

    def convert_files(
        self,
        files: List[Path],
        rollout_dir: Path,
        graph_dir: Path,
    ) -> None:
        if not files:
            return

        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        images_dir = graph_dir / "images"
        images_dir.mkdir(exist_ok=True)

        reward_threshold = float(self.config.get("reward_threshold", 0.5))

        # Load existing filtered trajs (for incremental merging)
        filtered_path = graph_dir / "filtered_trajs.json"
        if filtered_path.exists():
            with open(filtered_path, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        new_count = 0
        for fpath in files:
            step_idx = int(fpath.stem) if fpath.stem.isdigit() else 0

            with open(fpath, "r") as f:
                for line_idx, raw_line in enumerate(f):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    reward = _get_reward(data)
                    if reward < reward_threshold:
                        continue

                    messages = _parse_chatml(data)
                    if len(messages) < 3:
                        continue

                    # Copy images for this trajectory
                    image_base = rollout_dir / f"image_{step_idx}" / f"images_{line_idx}"
                    copied_images = self._copy_traj_images(
                        messages, image_base, images_dir,
                        step_idx, line_idx,
                    )

                    existing.append({
                        "messages": messages,
                        "image_paths": copied_images,
                        "reward": reward,
                    })
                    new_count += 1

        # Save merged filtered trajs
        with open(filtered_path, "w") as f:
            json.dump(existing, f, ensure_ascii=False)

        logger.info(
            "[SelfBootFilter] Added %d high-reward trajs (threshold=%.2f), "
            "total=%d → %s",
            new_count, reward_threshold, len(existing), filtered_path,
        )

    @staticmethod
    def _copy_traj_images(
        messages: List[Dict[str, str]],
        image_base: Path,
        images_dir: Path,
        step_idx: int,
        line_idx: int,
    ) -> List[str]:
        """Copy trajectory images to graph_dir/images/ and return relative paths."""
        copied: List[str] = []
        global_img_idx = 0

        for msg in messages:
            if msg["role"] != "user":
                continue
            num_images = len(_IMAGE_TAG_RE.findall(msg["content"]))
            for i in range(num_images):
                img_idx = global_img_idx + i
                src = None
                for suffix in (".png", ".jpg"):
                    candidate = image_base / f"{img_idx}{suffix}"
                    if candidate.exists():
                        src = candidate
                        break

                if src is not None:
                    dst_name = f"traj_{step_idx}_{line_idx}_{img_idx}{src.suffix}"
                    dst = images_dir / dst_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    copied.append(f"images/{dst_name}")
                else:
                    copied.append("")  # placeholder for missing image

            global_img_idx += num_images

        return copied
