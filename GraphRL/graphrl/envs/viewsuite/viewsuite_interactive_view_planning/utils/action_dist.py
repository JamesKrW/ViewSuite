"""Compute per-action distribution from a graph.json file.

Usage:
    python -m graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.action_dist \
        /path/to/graph/folder

    # custom output path
    python -m graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.action_dist \
        /path/to/graph/folder --output /tmp/action_dist.json
"""

import json
from collections import Counter
from pathlib import Path

import fire


def main(graph_dir: str, output: str | None = None):
    """Compute action distribution from graph edges.

    Args:
        graph_dir: Path to the graph folder containing graph.json.
        output: Output JSON path. Defaults to <graph_dir>/action_dist.json.
    """
    graph_dir = Path(graph_dir)
    graph_path = graph_dir / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"graph.json not found in {graph_dir}")

    with open(graph_path) as f:
        graph = json.load(f)

    edges = graph.get("edges", [])
    action_counter: Counter = Counter()
    total_actions = 0

    for edge in edges:
        obs_str = edge.get("obs_str", "")
        actions = [a.strip() for a in obs_str.split("|") if a.strip()]
        for action in actions:
            action_counter[action] += 1
            total_actions += 1

    # Sort by count descending
    sorted_actions = sorted(action_counter.items(), key=lambda x: -x[1])

    result = {
        "total_edges": len(edges),
        "total_actions": total_actions,
        "unique_actions": len(action_counter),
        "distribution": {
            action: {"count": count, "ratio": round(count / total_actions, 4)}
            for action, count in sorted_actions
        },
    }

    out_path = Path(output) if output else graph_dir / "action_dist.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total edges: {result['total_edges']}")
    print(f"Total actions: {result['total_actions']}")
    print(f"Unique actions: {result['unique_actions']}")
    print()
    for action, info in result["distribution"].items():
        print(f"  {action:<25s} {info['count']:>6d}  ({info['ratio']:.2%})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
