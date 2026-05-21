#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic scene split + within-scene sampling.

- Disjoint scene assignment to train/dev/test using Largest Remainder on a
  sorted scene list + a local RNG seeded by --seed.
- Within each split, per-scene sampling uses a stable per-scene RNG derived
  from SHA1(scene_name) to avoid dependence on PYTHONHASHSEED.
- "Sample" = immediate subdir under a scene that contains `meta.json`.
  If none, the whole scene is treated as a single sample.
- Output three sibling folders mirroring source structure:
  <src_root>_train, <src_root>_dev, <src_root>_test
- Link mode defaults to "hardlink" to avoid copying large data.
- Writes a manifest JSON listing selected scenes and samples for reproducibility.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import random

SCENE_REGEX = re.compile(r"^scene\d{4}_\d{2}$")
META_FILENAME = "meta.json"  # change if different


@dataclass
class SplitCfg:
    scene_split: float  # proportion of scenes for the split
    ratio: float        # proportion of samples kept per scene


def _stable_int_from_name(name: str, modulo: int = 10**9 + 7) -> int:
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(h, 16) % modulo


def _sorted_dirs(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def find_scene_dirs(src_root: Path) -> List[Path]:
    """Return immediate subdirs that look like ScanNet scenes (sorted)."""
    scenes = [p for p in _sorted_dirs(src_root) if SCENE_REGEX.match(p.name)]
    if not scenes:  # fallback: all immediate subdirs
        scenes = _sorted_dirs(src_root)
    return scenes


def list_samples_in_scene(scene_dir: Path) -> List[Path]:
    """
    A "sample" is an immediate child directory containing META_FILENAME.
    If none, treat the whole scene as one sample.
    """
    candidates = []
    for sub in _sorted_dirs(scene_dir):
        if (sub / META_FILENAME).exists():
            candidates.append(sub)
    return candidates if candidates else [scene_dir]


def allocate_scenes_by_ratio(
    scenes: List[Path],
    splits: List[str],
    scene_props: List[float],
    seed: int,
) -> Dict[str, List[Path]]:
    """
    Disjointly assign scenes to splits using Largest Remainder on counts.
    Deterministic: scenes are sorted, then shuffled by a local RNG with `seed`.
    """
    total = len(scenes)
    if total == 0:
        raise ValueError("No scene directories found.")
    if any(x < 0 for x in scene_props) or sum(scene_props) <= 0:
        raise ValueError(f"Invalid scene proportions: {scene_props}")

    # targets and base counts
    tot_prop = float(sum(scene_props))
    targets = [total * (p / tot_prop) for p in scene_props]
    base = [int(math.floor(t)) for t in targets]
    remainder = [t - b for t, b in zip(targets, base)]
    missing = total - sum(base)

    # deterministic tie-break on remainders
    order = sorted(range(len(splits)), key=lambda i: remainder[i], reverse=True)
    for i in range(missing):
        base[order[i % len(splits)]] += 1

    # deterministic shuffle (sorted input + fixed seed)
    rng = random.Random(seed)
    shuffled = scenes[:]  # copy
    rng.shuffle(shuffled)

    out: Dict[str, List[Path]] = {s: [] for s in splits}
    idx = 0
    for sname, count in zip(splits, base):
        out[sname] = shuffled[idx: idx + count]
        idx += count
    assert sum(len(v) for v in out.values()) == total
    return out


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src_file: Path, dst_file: Path, mode: str):
    if dst_file.exists():
        return
    if mode == "copy":
        shutil.copy2(src_file, dst_file)
    elif mode == "hardlink":
        os.link(src_file, dst_file)
    elif mode == "symlink":
        os.symlink(src_file, dst_file)
    else:
        raise ValueError(f"Unknown link_mode: {mode}")


def copy_entry(src: Path, dst: Path, mode: str):
    """
    Copy/link one file or directory. For directories, reproduce the tree in a
    sorted walk to keep deterministic order.
    """
    if src.is_dir():
        for root, dirs, files in os.walk(src):
            # sort to keep deterministic traversal
            dirs.sort()
            files.sort()
            rel = Path(root).relative_to(src)
            cur_dst = dst / rel
            cur_dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                sfile = Path(root) / f
                dfile = cur_dst / f
                link_or_copy_file(sfile, dfile, mode)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy_file(src, dst, mode)


def copy_scene_level_files(scene_dir: Path, dst_scene_root: Path, mode: str):
    """Copy/link files that live directly under the scene root (e.g. renders)."""
    for item in sorted(scene_dir.iterdir(), key=lambda p: p.name):
        if item.is_file():
            dst = dst_scene_root / item.name
            link_or_copy_file(item, dst, mode)


def main():
    ap = argparse.ArgumentParser(description="Deterministic scene split + within-scene sampling.")
    ap.add_argument("--src_root", type=str, required=True,
                    help="Source root (e.g., /home/.../scannet_proxy_task)")
    # split configs
    ap.add_argument("--train.scene_split", type=float, default=0.8)
    ap.add_argument("--train.ratio",       type=float, default=0.1)
    ap.add_argument("--dev.scene_split",   type=float, default=0.1)
    ap.add_argument("--dev.ratio",         type=float, default=0.1)
    ap.add_argument("--test.scene_split",  type=float, default=0.1)
    ap.add_argument("--test.ratio",        type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link_mode", type=str, default="hardlink",
                    choices=["copy", "hardlink", "symlink"],
                    help="How to materialize selected data.")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"src_root not found: {src_root}")

    # output dirs
    out_train = src_root.with_name(src_root.name + "_train")
    out_dev   = src_root.with_name(src_root.name + "_dev")
    out_test  = src_root.with_name(src_root.name + "_test")
    for p in (out_train, out_dev, out_test):
        ensure_dir(p)

    # collect scenes (sorted)
    scenes = find_scene_dirs(src_root)
    if not scenes:
        raise RuntimeError(f"No scene folders under {src_root}")

    # deterministic scene allocation
    splits = ["train", "dev", "test"]
    scene_props = [
        args.__dict__["train.scene_split"],
        args.__dict__["dev.scene_split"],
        args.__dict__["test.scene_split"],
    ]
    alloc = allocate_scenes_by_ratio(scenes, splits, scene_props, seed=args.seed)

    cfg_map: Dict[str, SplitCfg] = dict(
        train=SplitCfg(args.__dict__["train.scene_split"], args.__dict__["train.ratio"]),
        dev=  SplitCfg(args.__dict__["dev.scene_split"],   args.__dict__["dev.ratio"]),
        test= SplitCfg(args.__dict__["test.scene_split"],  args.__dict__["test.ratio"]),
    )
    out_map = dict(train=out_train, dev=out_dev, test=out_test)

    manifest = {
        "seed": args.seed,
        "link_mode": args.link_mode,
        "src_root": str(src_root),
        "splits": {k: asdict(v) for k, v in cfg_map.items()},
        "allocation": {},
        "samples": {},
    }

    # per split processing
    for split_name, scene_list in alloc.items():
        out_dir = out_map[split_name]
        ratio = cfg_map[split_name].ratio

        manifest["allocation"][split_name] = [p.name for p in scene_list]
        manifest["samples"][split_name] = {}

        for scene_dir in scene_list:
            samples = list_samples_in_scene(scene_dir)

            # whole-scene case
            if len(samples) == 1 and samples[0] == scene_dir:
                dst_scene_root = out_dir / scene_dir.name
                copy_entry(scene_dir, dst_scene_root, args.link_mode)
                manifest["samples"][split_name][scene_dir.name] = [scene_dir.name]
                continue

            # per-scene deterministic RNG based on SHA1(scene_name) + seed
            per_scene_seed = args.seed + _stable_int_from_name(scene_dir.name)
            rng = random.Random(per_scene_seed)

            n_total = len(samples)
            k = int(round(n_total * ratio))
            k = max(1, min(k, n_total))
            picked = sorted(rng.sample(samples, k), key=lambda p: p.name)

            dst_scene_root = out_dir / scene_dir.name
            dst_scene_root.mkdir(parents=True, exist_ok=True)

            picked_names = []
            for sdir in picked:
                rel = sdir.relative_to(scene_dir)  # one-level name
                dst = dst_scene_root / rel
                copy_entry(sdir, dst, args.link_mode)
                picked_names.append(rel.as_posix())

            copy_scene_level_files(scene_dir, dst_scene_root, args.link_mode)

            manifest["samples"][split_name][scene_dir.name] = picked_names

    # write manifest next to outputs
    manifest_path = src_root.parent / f"{src_root.name}_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
