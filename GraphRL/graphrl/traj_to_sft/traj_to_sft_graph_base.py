"""
Graph-flavored TrajToSFT template.

Many TrajToSFT subclasses follow the same pattern:

    1. Build (or reuse a cached) graph from VAGEN rollouts.
    2. Load the graph.
    3. Generate one or more LLaMA-Factory datasets from the graph.
    4. Write ``dataset_info.json`` + per-dataset .json files.

``TrajToSFTGraphBase`` codifies that template. Subclasses provide:

  * ``graph_builder_class()``   — return the ``VagenGraphBuilder`` subclass
    that knows how to convert this env's rollouts into a ``BaseGraph``.
  * ``generate_datasets(graph, images_dir)`` — produce the
    ``{name: (records, fmt_override)}`` mapping.

Subclasses that need to source rollouts from somewhere other than
``self.paths.rollout_data`` (e.g. ``RandomActionTrajToSFT`` which runs an eval
subprocess first) override ``_build_or_load_graph()`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from graphrl.traj_to_sft.traj_to_sft_base import TrajToSFTModule
from graphrl.traj_to_sft.utils.base_graph import BaseGraph
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder

logger = logging.getLogger(__name__)


class TrajToSFTGraphBase(TrajToSFTModule):
    """Graph-based TrajToSFT template. Subclass this for graph-driven envs."""

    name = "TrajToSFT(graph)"

    # ── ``run()`` template ────────────────────────────────────────────────

    def run(self) -> None:
        graph_dir = self._build_or_load_graph()
        graph = BaseGraph.load(graph_dir)
        self._log(f"Graph loaded: {graph.num_nodes} nodes, {graph.num_edges} edges")
        if graph.num_edges == 0:
            raise RuntimeError(f"Graph at {graph_dir} is empty.")
        datasets = self.generate_datasets(graph, graph_dir / "images")
        self.write_dataset_info(datasets)

    # ── subclass interface (required) ─────────────────────────────────────

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        """Return the ``VagenGraphBuilder`` subclass to use for this env."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement graph_builder_class()."
        )

    def generate_datasets(
        self,
        graph: BaseGraph,
        images_dir: Path,
    ) -> Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """Return ``{dataset_name: (records, fmt_override)}``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_datasets()."
        )

    # ── default graph build (override for non-rollout-based sources) ──────

    @property
    def graph_dir(self) -> Path:
        """Where the built graph lives. Default: ``iter_XXX/traj_to_sft/graph/``.

        The graph is a TrajToSFT artefact (built from RL's rollouts but owned
        by this phase), so it lives under ``traj_to_sft/`` alongside other
        TrajToSFT scratch (``sft_data_old``, ``reasoning_dump``) — keeping
        ``rl/`` as a clean home for VAGEN-only outputs.
        """
        return self.paths.base_dir / "traj_to_sft" / "graph"

    def _build_or_load_graph(self) -> Path:
        """Build a graph from ``self.paths.rollout_data``, or reuse the cache."""
        graph_dir = self.graph_dir
        graph_json = graph_dir / "graph.json"

        rollouts = sorted(self.paths.rollout_data.glob("*.jsonl"))
        if rollouts:
            graph_dir.mkdir(parents=True, exist_ok=True)
            builder_cfg = self.config.get("graph_builder", {}) or {}
            builder = self.graph_builder_class()(builder_cfg)
            self._log(f"Building graph from {len(rollouts)} rollout file(s)")
            builder.convert_files(rollouts, self.paths.rollout_data, graph_dir)
        elif graph_json.exists():
            self._log(f"No new rollouts; using cached graph at {graph_json}")
        else:
            raise RuntimeError(
                f"No rollouts at {self.paths.rollout_data} and no cached graph "
                f"at {graph_json}."
            )
        return graph_dir
