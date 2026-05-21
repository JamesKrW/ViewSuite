# Sokoban Text Environment

Reference implementation of a **graph-based TrajToSFT subclass** in GraphRL.
Text-only Sokoban: VAGEN rollouts in, four LLaMA-Factory datasets out
(`path_to_view_direct`, `path_to_view_mcq`, `view_to_path`,
`state_reachable`).

## What you write to add a new env

A new env that follows the standard "build graph from rollouts → sample
records from graph" pattern needs **two classes**:

| Class | Base | What you implement |
|---|---|---|
| `MyEnvGraphBuilder` | `VagenGraphBuilder` | `traj_to_transitions()` |
| `MyEnvTrajToSFT` | `TrajToSFTGraphBase` | `graph_builder_class()` + `generate_datasets()` |

Everything else (multiprocessing, image copying, dedup, graph caching,
`dataset_info.json` writing, resume-safe cleanup, HF upload, phase
materialisation when a phase is skipped) is handled by the framework.

For non-graph patterns (random-eval subprocess, fixed-dataset symlink, etc.)
extend `TrajToSFTModule` directly — see
`graphrl/envs/viewsuite/viewsuite_random_sft_rl/` and
`graphrl/envs/viewsuite/viewsuite_sft_rl/` for examples.

---

## The graph format

Every graph-based TrajToSFT pipeline persists a `BaseGraph` to
`{iter_dir}/traj_to_sft/graph/graph.json`:

```json
{
  "nodes": {
    "<16-hex-key>": {
      "state":       <any JSON>,
      "obs_str":     "LLM-readable string, may contain <image> placeholders",
      "image_paths": ["images/abc.jpg"],
      "extra":       {}
    }
  },
  "edges": [
    {"from": "<key>", "to": "<key>", "obs_str": "action text",
     "image_paths": [], "extra": {}}
  ]
}
```

- `state` — raw state, used only for dedup (`sha256` key by default).
- `obs_str` — string shown to the LLM in SFT prompts.
- `image_paths` — relative paths inside `graph/images/` (auto-copied).

For text-only envs, `state == obs_str` (the grid string). For multimodal
envs see `viewsuite_interactive_view_planning`.

---

## Three layers of node/edge data

Dedup logic lives on the data classes, not the graph builder.

### Layer 0 — abstract base ([`base_graph.py`](../../../traj_to_sft/utils/base_graph.py))

```python
class NodeData(ABC):
    def unique_key(self) -> str: ...        # deterministic ID
    def bucket_key(self) -> str: ...        # coarse grouping
    def is_similar_to(self, other) -> bool: ...

class EdgeData(ABC):
    def unique_key(self) -> str: ...
    def bucket_key(self) -> str: ...
    def is_similar_to(self, other) -> bool: ...
```

### Layer 1 — VAGEN defaults ([`graph_builder.py`](../../../traj_to_sft/utils/graph_builder.py))

```python
class VagenNodeData(NodeData):
    # Fields: state, obs_str, source_images, image_paths, extra
    # unique_key = sha256(state)[:16];  bucket_key = unique_key;  is_similar_to = True

class VagenEdgeData(EdgeData):
    # Fields: obs_str, image_paths, extra
    # unique_key = repr(obs_str);  bucket_key = unique_key;  is_similar_to = obs_str equality
```

### Layer 2 — env-specific (override only what you need to)

```python
# viewsuite_interactive_view_planning: scene-bucket + pose-tolerance dedup
class ViewSuiteNodeData(VagenNodeData):
    def unique_key(self):  return md5(scene_id + pose_at_4dp)
    def bucket_key(self):  return self.state["scene_id"]
    def is_similar_to(self, other):  return pose_distance < 0.05  # m
```

---

## Graph builder — implement `traj_to_transitions()`

```python
# graphrl/envs/sokoban/sokoban_text/sokoban_graph_builder.py
from graphrl.traj_to_sft.utils.graph_builder import (
    VagenGraphBuilder, VagenNodeData, VagenEdgeData,
)

class SokobanTextGraphBuilder(VagenGraphBuilder):
    """Text-only Sokoban — uses default VagenNodeData/VagenEdgeData."""

    def traj_to_transitions(self, messages, rollout_dir, step_idx, line_idx):
        """
        Args:
            messages:    [{"role": "user"|"assistant", "content": str}, ...]
                         Parsed from VAGEN rollout JSONL (input + output combined).
            rollout_dir: Path to the rollout dir (for locating image files).
            step_idx:    int from the JSONL filename.
            line_idx:    0-based line index within the JSONL file.

        Returns:
            List[(src_node, edge, dst_node)] — triples of NodeData / EdgeData.
        """
        transitions = []
        current_state, pending_action = None, None
        for msg in messages:
            if msg["role"] == "user":
                grid = parse_grid(msg["content"])
                if grid is not None:
                    if current_state is not None and pending_action is not None:
                        transitions.append((
                            VagenNodeData(state=current_state, obs_str=current_state),
                            VagenEdgeData(obs_str=pending_action),
                            VagenNodeData(state=grid, obs_str=grid),
                        ))
                    current_state, pending_action = grid, None
            elif msg["role"] == "assistant":
                action = parse_action(msg["content"])
                if action:
                    pending_action = action
        return transitions
```

`VagenGraphBuilder` does the rest: VAGEN JSONL parsing, multiprocessing
(`num_workers` config key), image copying, dedup, incremental
`graph.json` saves.

If you subclass `VagenNodeData` / `VagenEdgeData`, override
`_make_node_data()` / `_make_edge_data()` so dedup reconstructs the
correct class:

```python
class MyEnvGraphBuilder(VagenGraphBuilder):
    def _make_node_data(self, ndata):
        return MyNodeData(state=ndata["state"], obs_str=ndata.get("obs_str"), ...)
```

---

## TrajToSFT subclass — implement two methods

```python
# graphrl/envs/sokoban/sokoban_text/traj_to_sft.py
import random
from typing import Any, Dict, List, Optional, Tuple, Type

from graphrl import TrajToSFTGraphBase
from graphrl.traj_to_sft.utils.base_graph import BaseGraph
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder
from graphrl.envs.sokoban.sokoban_text.sokoban_graph_builder import SokobanTextGraphBuilder

_SHAREGPT_FMT = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {"role_tag": "role", "content_tag": "content",
             "user_tag": "user", "assistant_tag": "assistant",
             "system_tag": "system"},
}


class SokobanTextTrajToSFT(TrajToSFTGraphBase):
    """Sokoban-text rollouts → 4 LLaMA-Factory datasets."""

    name = "TrajToSFT(sokoban_text)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return SokobanTextGraphBuilder

    def generate_datasets(
        self, graph: BaseGraph, images_dir,
    ) -> Dict[str, Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        from .utils.sft_generators import (
            generate_path_to_view_direct,
            generate_path_to_view_mcq,
            generate_view_to_path,
            generate_state_reachable,
        )
        cfg = self.config
        rng = random.Random(cfg.get("seed", 42))

        result = {}
        # …call each generator, stuff records + format-override into result…
        return result
```

`TrajToSFTGraphBase.run()` handles:

1. Build (or reuse cached) graph from `self.paths.rollout_data` via the
   class returned by `graph_builder_class()`.
2. Load the graph from `iter_dir/traj_to_sft/graph/`.
3. Call your `generate_datasets(graph, images_dir)`.
4. Write `dataset_info.json` + per-dataset JSON via
   `self.write_dataset_info()`.

### `generate_datasets()` return contract

```python
Dict[name, (records, fmt_override_or_None)]
```

| `fmt_override` | Meaning |
|---|---|
| `None` | auto-detect (Alpaca if records have `instruction`; else look at first record) |
| `dict` | merged verbatim into the `dataset_info.json` entry — set `{"formatting": "sharegpt", "columns": {...}}` for ShareGPT multi-turn |

---

## Sokoban-specific implementation

### Graph builder ([`sokoban_graph_builder.py`](sokoban_graph_builder.py))

Extracts grid states from `[Initial Observation]` / `After that, the
observation is:` blocks and actions from `<answer>…</answer>` tags.

Uses **default dedup** (VagenNodeData) — two grids are the same node iff
their normalised text matches; two edges are the same iff
`repr(action_text)` matches. Failed agent outputs are skipped but the
subsequent state is still tracked, keeping later transitions aligned.

### SFT generator ([`traj_to_sft.py`](traj_to_sft.py) + [`utils/sft_generators.py`](utils/sft_generators.py))

| Dataset | Task |
|---|---|
| `path_to_view_direct` | `(state, actions)` → `<prediction>final_state</prediction>` |
| `path_to_view_mcq` | `(state, actions, 4 options)` → correct letter A–D |
| `view_to_path` | `(state, next_state)` → action sequence |
| `state_reachable` | multi-turn navigation from `s0` to `sN` |

All datasets use ShareGPT message format. Path-based datasets sample
from the graph via `graph.sample_paths()`; `view_to_path` samples
individual edges.

### Pipeline.yaml

```yaml
general_overrides:
  traj_to_sft:
    module: graphrl.envs.sokoban.sokoban_text.SokobanTextTrajToSFT
    generators:
      - path_to_view_direct
      - path_to_view_mcq
      - view_to_path
      - state_reachable
    path_to_view: { min_path_len: 1, max_path_len: 3, num_samples: 2000 }
    view_to_path: { num_samples: 2000 }
    state_reachable:  { min_path_len: 3, max_path_len: 5, num_samples: 1000 }
    seed: 42
```

---

## Useful `BaseGraph` methods

```python
graph = BaseGraph.load(graph_dir)

# Sample random paths (returns steps with from_id/from_state/action/to_id/to_state)
paths = graph.sample_paths(min_len=1, max_len=3, num_samples=500, rng=rng)

# Get random node texts (for MCQ negatives, etc.)
texts = graph.get_random_state_texts(n=3, exclude_ids={"abc123"}, rng=rng)

# Get random node attribute dicts (with optional filtering)
nodes = graph.get_random_nodes(n=3, exclude_ids={"abc123"},
                               rng=rng, filter_fn=lambda nd: nd["scene_id"] == s)

# Direct NetworkX access
nx_graph = graph.to_networkx()  # returns the underlying nx.MultiDiGraph
```
