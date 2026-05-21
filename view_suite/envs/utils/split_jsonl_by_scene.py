from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# -------------------------------
# divide by scene: train/eval/test
# -------------------------------
# --- Stable split: sort scenes and keep deterministic iteration order ---
def split_jsonl_by_scene(
    jsonl_path: str,
    ratios: Tuple[float, float, float] = (80, 10, 10),  # (train, eval, test)
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    in_path = Path(jsonl_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    scene_to_items: Dict[str, List[dict]] = {}
    total_samples = 0
    with in_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {ln} invalid JSON: {e}") from e
            if "scene_id" not in obj:
                raise KeyError(f"Line {ln} missing key 'scene_id'")
            sid = str(obj["scene_id"])
            scene_to_items.setdefault(sid, []).append(obj)
            total_samples += 1

    # <<< IMPORTANT: sort scenes for stability >>>
    scenes = sorted(scene_to_items.keys(), key=str)
    total_scenes = len(scenes)
    if total_scenes == 0:
        raise ValueError("No scenes found.")

    r_train, r_eval, r_test = ratios
    if any(r < 0 for r in (r_train, r_eval, r_test)) or (r_train + r_eval + r_test) <= 0:
        raise ValueError(f"Invalid ratios: {ratios}")

    R = [float(r_train), float(r_eval), float(r_test)]
    R_sum = sum(R)
    target = [total_scenes * r / R_sum for r in R]
    base = [int(x) for x in target]
    remainder = [t - b for t, b in zip(target, base)]
    missing = total_scenes - sum(base)

    # Largest Remainder with a stable tie-break (index order is deterministic)
    order = sorted(range(3), key=lambda i: remainder[i], reverse=True)
    for i in range(missing):
        base[order[i % 3]] += 1

    n_train, n_eval, n_test = base
    assert n_train + n_eval + n_test == total_scenes

    # <<< IMPORTANT: use a local RNG with fixed seed >>>
    rng = random.Random(seed)
    scenes_shuffled = scenes[:]          # copy
    rng.shuffle(scenes_shuffled)         # deterministic given sorted input

    # <<< IMPORTANT: KEEP LISTS (not sets) to preserve order >>>
    train_sids = scenes_shuffled[:n_train]
    eval_sids  = scenes_shuffled[n_train:n_train + n_eval]
    test_sids  = scenes_shuffled[n_train + n_eval:]

    out_dir = Path(output_dir) if output_dir is not None else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem
    out_paths = {
        "train": out_dir / f"{stem}_train.jsonl",
        "eval":  out_dir / f"{stem}_eval.jsonl",
        "test":  out_dir / f"{stem}_test.jsonl",
    }

    counts_samples = {"train": 0, "eval": 0, "test": 0}

    # <<< IMPORTANT: dump splits in a stable scene-id order >>>
    def dump_split(split_name: str, sid_list: List[str]) -> None:
        p = out_paths[split_name]
        with p.open("w", encoding="utf-8") as wf:
            for sid in sid_list:
                # keep original per-scene item order as in source file
                for obj in scene_to_items[sid]:
                    wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    counts_samples[split_name] += 1

    dump_split("train", train_sids)
    dump_split("eval",  eval_sids)
    dump_split("test",  test_sids)

    summary = {
        "paths": {k: str(v) for k, v in out_paths.items()},
        "scenes": {
            "total": total_scenes,
            "train": len(train_sids),
            "eval":  len(eval_sids),
            "test":  len(test_sids),
        },
        "samples": {
            "total": total_samples,
            "train": counts_samples["train"],
            "eval":  counts_samples["eval"],
            "test":  counts_samples["test"],
        },
    }
    return summary


# --- Stable mini: deterministic subset given fixed file order & seed ---
def sample_jsonl_mini(
    jsonl_path: str,
    ratio: float = 0.1,
    output_dir: Optional[str] = None,
    seed: int = 42,
    keep_at_least: int = 1,
) -> Dict[str, int]:
    in_path = Path(jsonl_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    lines: List[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for s in f:
            s = s.rstrip("\n")
            if s.strip():
                lines.append(s)

    total = len(lines)
    if total == 0:
        raise ValueError("Empty JSONL.")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")

    rng = random.Random(seed)   # local RNG with fixed seed
    if ratio >= 1.0:
        kept_idx = list(range(total))
    else:
        k = int(total * ratio)
        if k == 0:
            k = min(total, max(keep_at_least, 1))
        kept_idx = sorted(rng.sample(range(total), k))  # order results

    out_dir = Path(output_dir) if output_dir is not None else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem
    out_path = out_dir / f"{stem}_mini.jsonl"

    with out_path.open("w", encoding="utf-8") as wf:
        for i in kept_idx:
            wf.write(lines[i] + "\n")

    return {"total": total, "kept": len(kept_idx), "path": str(out_path)}

# -------------------------------
# Fire CLI
# -------------------------------
if __name__ == "__main__":
    import fire
    fire.Fire({
        "split": split_jsonl_by_scene,   # python script.py split --jsonl_path=xxx.jsonl --ratios="(80,10,10)" --seed=42
        "mini":  sample_jsonl_mini,      # python script.py mini  --jsonl_path=xxx.jsonl --ratio=0.1 --seed=42
    })
