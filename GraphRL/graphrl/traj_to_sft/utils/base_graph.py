"""
Generic directed graph for RL trajectory data.

Backed by a NetworkX MultiDiGraph.

Disk format  (graph_dir/graph.json):
  {
    "nodes": {
      "<key>": {
        "state":       <any JSON-serializable>,
        "obs_str":     "<text representation, may include <image> placeholders>",
        "image_paths": ["images/abc.jpg", ...],
        "extra":       {...}
      }
    },
    "edges": [
      {"from": "<key>", "to": "<key>", "obs_str": "<action text>",
       "image_paths": [...], "extra": {...}}
    ]
  }

Keys are arbitrary strings (typically sha256 of state).
Images are stored in  graph_dir/images/  (optional, for multimodal environments).
"""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


# ── Abstract data interfaces ────────────────────────────────────────────────

class NodeData(ABC):
    """
    Abstract base for graph node data.

    Concrete subclasses store whatever fields the environment needs
    (state, obs_str, image_paths, extra, ...) and implement the
    three-method dedup interface used by the graph builder.
    """

    @abstractmethod
    def unique_key(self) -> str:
        """Deterministic unique ID stored in the graph."""
        ...

    @abstractmethod
    def bucket_key(self) -> str:
        """Coarse grouping key for dedup scan."""
        ...

    @abstractmethod
    def is_similar_to(self, other: "NodeData") -> bool:
        """Whether this node is a duplicate of *other* (within same bucket)."""
        ...


class EdgeData(ABC):
    """
    Abstract base for graph edge data.

    Dedup for edges operates within the (from_uid, to_uid) pair first,
    then uses bucket_key + is_similar_to, same as node dedup.
    """

    @abstractmethod
    def unique_key(self) -> str:
        """Deterministic unique ID for this edge (scoped to src/dst pair)."""
        ...

    @abstractmethod
    def bucket_key(self) -> str:
        """Coarse grouping key for dedup scan."""
        ...

    @abstractmethod
    def is_similar_to(self, other: "EdgeData") -> bool:
        """Whether this edge is a duplicate of *other* (within same bucket and src/dst pair)."""
        ...


# ── BaseGraph ─────────────────────────────────────────────────────────────────

class BaseGraph:
    """
    Generic in-memory directed graph backed by a NetworkX MultiDiGraph.

    Nodes store:  state, obs_str, image_paths, extra
    Edges store:  obs_str, image_paths, extra

    Entry-point for adding data:
      _upsert_node / _add_edge  (used by graph builders with pre-computed keys)
    """

    GRAPH_FILE = "graph.json"

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ── node access (dict-like proxy to NetworkX NodeView) ────────────────────

    @property
    def nodes(self):
        """Dict-like access to node attributes: graph.nodes[key]["obs_str"]."""
        return self._g.nodes

    # ── mutation (caller supplies key / eid) ─────────────────────────────────

    def _upsert_node(
        self,
        key: str,
        state: Any,
        obs_str: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> None:
        """Add a node with a pre-computed key (no-op if key already exists)."""
        if key not in self._g:
            self._g.add_node(
                key,
                state=state,
                obs_str=obs_str if obs_str is not None else str(state),
                image_paths=image_paths or [],
                extra=extra or {},
            )

    def _add_edge(
        self,
        eid: str,
        from_key: str,
        obs_str: str,
        to_key: str,
        extra: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> None:
        """Add an edge identified by eid (no-op if eid already exists)."""
        if not self._g.has_edge(from_key, to_key, key=eid):
            self._g.add_edge(
                from_key, to_key, key=eid,
                obs_str=obs_str,
                image_paths=image_paths or [],
                extra=extra or {},
            )

    # ── merge ─────────────────────────────────────────────────────────────────

    def merge_from(self, other: "BaseGraph") -> None:
        """Merge all nodes and edges from *other* into self (first-writer wins)."""
        for key, ndata in other._g.nodes(data=True):
            if key not in self._g:
                self._g.add_node(key, **ndata)

        for u, v, eid, data in other._g.edges(data=True, keys=True):
            if not self._g.has_edge(u, v, key=eid):
                # Ensure both endpoint nodes exist
                for nk in (u, v):
                    if nk not in self._g and nk in other._g:
                        self._g.add_node(nk, **other._g.nodes[nk])
                self._g.add_edge(u, v, key=eid, **data)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, graph_dir: Path) -> None:
        """
        Save to graph_dir/graph.json, merging with any existing graph first.
        Safe to call repeatedly as new rollout batches arrive.
        """
        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_file = graph_dir / self.GRAPH_FILE

        if graph_file.exists():
            existing = type(self).load(graph_dir)
            existing.merge_from(self)
            target = existing
        else:
            target = self

        nodes_out = {k: dict(v) for k, v in target._g.nodes(data=True)}
        edges_out = [
            {
                "from": u, "to": v,
                "obs_str": d["obs_str"],
                "image_paths": d.get("image_paths", []),
                "extra": d.get("extra", {}),
            }
            for u, v, _eid, d in target._g.edges(data=True, keys=True)
        ]
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes_out, "edges": edges_out}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, graph_dir: Path) -> "BaseGraph":
        """Load from graph_dir/graph.json. Returns empty graph if not found."""
        g = cls()
        graph_file = Path(graph_dir) / cls.GRAPH_FILE
        if not graph_file.exists():
            return g
        with open(graph_file, encoding="utf-8") as f:
            data = json.load(f)
        for key, ndata in data.get("nodes", {}).items():
            # Backward compat: text → obs_str, images → image_paths
            if "text" in ndata and "obs_str" not in ndata:
                ndata["obs_str"] = ndata.pop("text")
            if "images" in ndata and "image_paths" not in ndata:
                ndata["image_paths"] = ndata.pop("images")
            g._g.add_node(key, **ndata)
        for edge in data.get("edges", []):
            obs_str = edge.get("obs_str", edge.get("text", ""))
            image_paths = edge.get("image_paths", [])
            g._g.add_edge(
                edge["from"], edge["to"], key=repr(obs_str),
                obs_str=obs_str,
                image_paths=image_paths,
                extra=edge.get("extra", {}),
            )
        return g

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for multiprocessing / pickling)."""
        return {
            "nodes": {k: dict(v) for k, v in self._g.nodes(data=True)},
            "edges": [
                {
                    "from": u, "to": v,
                    "obs_str": d["obs_str"],
                    "image_paths": d.get("image_paths", []),
                    "extra": d.get("extra", {}),
                }
                for u, v, _eid, d in self._g.edges(data=True, keys=True)
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseGraph":
        """Deserialize from the result of to_dict()."""
        g = cls()
        for key, ndata in d.get("nodes", {}).items():
            g._g.add_node(key, **ndata)
        for edge in d.get("edges", []):
            obs_str = edge.get("obs_str", edge.get("text", ""))
            g._g.add_edge(
                edge["from"], edge["to"], key=repr(obs_str),
                obs_str=obs_str,
                image_paths=edge.get("image_paths", []),
                extra=edge.get("extra", {}),
            )
        return g

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample_paths(
        self,
        min_len: int,
        max_len: int,
        num_samples: int,
        rng: Optional[random.Random] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Sample random walks without replacement (on path identity).

        Returns a list of paths; each path is a list of steps:
            [{"from_id": <key>, "from_state": <obs_str>, "action": <edge obs_str>,
              "to_id": <key>, "to_state": <obs_str>}, ...]
        """
        if rng is None:
            rng = random.Random()

        node_ids = list(self._g.nodes())
        if not node_ids or self._g.number_of_edges() == 0:
            return []

        seen: Set[Tuple] = set()
        paths: List[List[Dict[str, Any]]] = []
        max_attempts = num_samples * 30
        attempts = 0

        while len(paths) < num_samples and attempts < max_attempts:
            attempts += 1
            cur = rng.choice(node_ids)
            target_len = rng.randint(min_len, max_len)
            steps: List[Dict[str, Any]] = []
            ekey_seq: List[str] = []
            visited: Set[str] = {cur}
            ok = True

            for _ in range(target_len):
                out = list(self._g.out_edges(cur, data=True, keys=True))
                if not out:
                    ok = False
                    break
                # Prefer edges leading to unvisited nodes to avoid cycles
                unvisited_out = [e for e in out if e[1] not in visited]
                chosen_pool = unvisited_out if unvisited_out else None
                if chosen_pool is None:
                    # All neighbours already visited — stop walk early
                    break
                u, v, eid, data = rng.choice(chosen_pool)
                steps.append(
                    {
                        "from_id": u,
                        "from_state": self._g.nodes[u]["obs_str"],
                        "action": data["obs_str"],
                        "to_id": v,
                        "to_state": self._g.nodes[v]["obs_str"],
                    }
                )
                ekey_seq.append(eid)
                visited.add(v)
                cur = v

            if not ok or len(steps) < min_len:
                continue

            path_key = tuple(ekey_seq)
            if path_key in seen:
                continue
            seen.add(path_key)
            paths.append(steps)

        return paths

    def get_random_state_texts(
        self,
        n: int,
        exclude_ids: Set[str],
        rng: random.Random,
    ) -> List[str]:
        """Return up to *n* random node obs_str representations, excluding *exclude_ids*."""
        candidates = [k for k in self._g.nodes() if k not in exclude_ids]
        k = min(n, len(candidates))
        if k == 0:
            return []
        return [self._g.nodes[key]["obs_str"] for key in rng.sample(candidates, k)]

    def get_random_nodes(
        self,
        n: int,
        exclude_ids: Set[str],
        rng: random.Random,
        filter_fn: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Return up to *n* random node attribute dicts, excluding *exclude_ids*.

        Each dict has keys: state, obs_str, image_paths, extra (from the NetworkX node).
        Optional *filter_fn(node_attrs) -> bool* restricts candidates further
        (e.g. to nodes from the same scene).
        """
        candidates = []
        for k in self._g.nodes():
            if k in exclude_ids:
                continue
            attrs = dict(self._g.nodes[k])
            if filter_fn and not filter_fn(attrs):
                continue
            candidates.append((k, attrs))
        k = min(n, len(candidates))
        if k == 0:
            return []
        selected = rng.sample(candidates, k)
        return [{"id": key, **attrs} for key, attrs in selected]

    # ── backward-compat edge list (used by sft_generators) ───────────────────

    @property
    def _edges(self) -> List[Dict[str, Any]]:
        """Flat list of edge dicts for SFT generators that iterate edges directly."""
        return [
            {"from": u, "to": v, "obs_str": d["obs_str"], "extra": d.get("extra", {})}
            for u, v, _eid, d in self._g.edges(data=True, keys=True)
        ]

    # ── networkx access ───────────────────────────────────────────────────────

    def to_networkx(self) -> nx.MultiDiGraph:
        """Return the underlying NetworkX MultiDiGraph directly (no conversion)."""
        return self._g

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()
