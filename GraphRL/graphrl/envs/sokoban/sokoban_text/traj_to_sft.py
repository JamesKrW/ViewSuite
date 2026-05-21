"""
Sokoban-text TrajToSFT phase.

Reads VAGEN rollouts from ``paths.rollout_data``, builds a Sokoban graph via
``SokobanTextGraphBuilder``, then generates four LLaMA-Factory datasets:

  path_to_view_direct  — (state, action_seq) → <prediction>final_state</prediction>
  path_to_view_mcq     — (state, action_seq) → MCQ letter (A-D)
  view_to_path         — (state, next_state) → action sequence
  state_reachable          — multi-turn path navigation

All datasets use ShareGPT message format. Empty rollout dirs are tolerated
(the graph step is skipped if there's nothing new to convert and a cached
graph already exists at ``base_dir/traj_to_sft/graph``).

Pipeline.yaml::

    traj_to_sft:
      module: graphrl.envs.sokoban.sokoban_text.SokobanTextTrajToSFT
      generators: [...]
      path_to_view: { min_path_len: 1, max_path_len: 3, num_samples: 2000 }
      view_to_path: { num_samples: 2000 }
      state_reachable:  { min_path_len: 3, max_path_len: 5, num_samples: 1000 }
      seed: 42
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from graphrl import TrajToSFTGraphBase
from graphrl.traj_to_sft.utils.base_graph import BaseGraph
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder
from graphrl.envs.sokoban.sokoban_text.sokoban_graph_builder import SokobanTextGraphBuilder

logger = logging.getLogger(__name__)

_ALL_GENERATORS = [
    "path_to_view_direct",
    "path_to_view_mcq",
    "view_to_path",
    "state_reachable",
]

_SHAREGPT_FMT = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}


class SokobanTextTrajToSFT(TrajToSFTGraphBase):
    """Convert Sokoban-text VAGEN rollouts → 4 LLaMA-Factory datasets."""

    name = "TrajToSFT(sokoban_text)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return SokobanTextGraphBuilder

    def generate_datasets(
        self,
        graph: BaseGraph,
        images_dir: Path,
    ) -> Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        from graphrl.envs.sokoban.sokoban_text.utils.sft_generators import (
            generate_path_to_view_direct,
            generate_path_to_view_mcq,
            generate_view_to_path,
            generate_state_reachable,
        )

        cfg = self.config
        enabled: List[str] = cfg.get("generators", _ALL_GENERATORS)
        fwd_cfg: Dict[str, Any] = cfg.get("path_to_view", {})
        inv_cfg: Dict[str, Any] = cfg.get("view_to_path", {})
        reach_cfg: Dict[str, Any] = cfg.get("state_reachable", {})
        seed: int = cfg.get("seed", 42)

        master_rng = random.Random(seed)

        def child_rng() -> random.Random:
            return random.Random(master_rng.randint(0, 2 ** 32 - 1))

        result: Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}

        if "path_to_view_direct" in enabled:
            records = generate_path_to_view_direct(
                graph,
                min_path_len=fwd_cfg.get("min_path_len", 1),
                max_path_len=fwd_cfg.get("max_path_len", 3),
                num_samples=fwd_cfg.get("num_samples", 2000),
                rng=child_rng(),
            )
            result["path_to_view_direct"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] path_to_view_direct: %d", self.name, len(records))

        if "path_to_view_mcq" in enabled:
            records = generate_path_to_view_mcq(
                graph,
                min_path_len=fwd_cfg.get("min_path_len", 1),
                max_path_len=fwd_cfg.get("max_path_len", 3),
                num_samples=fwd_cfg.get("num_samples", 2000),
                rng=child_rng(),
            )
            result["path_to_view_mcq"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] path_to_view_mcq: %d", self.name, len(records))

        if "view_to_path" in enabled:
            records = generate_view_to_path(
                graph,
                num_samples=inv_cfg.get("num_samples", 2000),
                rng=child_rng(),
            )
            result["view_to_path"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] view_to_path: %d", self.name, len(records))

        if "state_reachable" in enabled:
            records = generate_state_reachable(
                graph,
                min_path_len=reach_cfg.get("min_path_len", 3),
                max_path_len=reach_cfg.get("max_path_len", 5),
                num_samples=reach_cfg.get("num_samples", 1000),
                rng=child_rng(),
            )
            result["state_reachable"] = (records, _SHAREGPT_FMT)
            logger.info("[%s] state_reachable: %d", self.name, len(records))

        return result
