import json
from pathlib import Path
from typing import Any, Dict, List


def count_lines(jsonl_path: Path) -> int:
    """Count lines once. For huge files, this is still a single sequential pass."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def read_jsonl_line_by_index(jsonl_path: Path, idx: int) -> Dict[str, Any]:
    """Sequentially scan until idx-th (0-based) line, return parsed JSON."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range for {jsonl_path}")

# -------- IO helpers --------
def load_jsonl_items(jsonl_path: Path) -> List[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def resolve_rel_image(jsonl_path: Path, rel: str,data_set_root:str=None) -> Path:
    if data_set_root is not None:
        base = data_set_root
    else:
        base = jsonl_path.parent
    if rel.startswith("./"):
        rel = rel[2:]
    return base / rel


