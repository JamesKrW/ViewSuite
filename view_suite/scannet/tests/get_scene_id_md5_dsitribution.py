#!/usr/bin/env python3
# test_scene_distribution.py
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
from collections import Counter
from typing import List


def md5_u64(text: str) -> int:
    d = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(d[:8], byteorder="big", signed=False)


def md5_mod_bucket(scene_id: str, k: int) -> int:
    return md5_u64(scene_id) % k


def hrw_best_bucket(scene_id: str, urls: List[str]) -> int:
    best_score = None
    best_idx = 0
    for i, u in enumerate(urls):
        s = md5_u64(f"{scene_id}|{u}")
        if best_score is None or s > best_score:
            best_score = s
            best_idx = i
    return best_idx


def list_scene_ids(scans_dir: str) -> List[str]:
    items = []
    for name in os.listdir(scans_dir):
        p = os.path.join(scans_dir, name)
        if os.path.isdir(p):
            items.append(name)
    items.sort()
    return items


def summarize_counts(title: str, counts: List[int]) -> None:
    n = sum(counts)
    k = len(counts)
    mean = n / k if k else 0.0
    mn = min(counts) if counts else 0
    mx = max(counts) if counts else 0
    std = statistics.pstdev(counts) if k > 0 else 0.0

    # Chi-square vs uniform expectation (rough indicator)
    chi2 = 0.0
    if mean > 0:
        for c in counts:
            chi2 += (c - mean) ** 2 / mean

    print(f"\n== {title} ==")
    print(f"total_scenes: {n}")
    print(f"buckets(k):   {k}")
    print(f"min:         {mn}")
    print(f"max:         {mx}")
    print(f"mean:        {mean:.2f}")
    print(f"std:         {std:.2f}")
    print(f"chi2:        {chi2:.2f}  (lower ~ more uniform)")
    print("counts:      " + " ".join(str(c) for c in counts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans_dir", required=True, help="Path to scannet scans folder containing scene subfolders")
    ap.add_argument("--urls", required=True, help="Semicolon-separated base URLs, e.g. http://a;http://b;http://c")
    args = ap.parse_args()

    scans_dir = args.scans_dir
    urls = [u.strip().rstrip("/") for u in args.urls.split(";") if u.strip()]
    if not urls:
        raise SystemExit("No urls provided")

    scene_ids = list_scene_ids(scans_dir)
    if not scene_ids:
        raise SystemExit(f"No scene subfolders found in: {scans_dir}")

    k = len(urls)

    md5_counts = Counter()
    hrw_counts = Counter()

    for s in scene_ids:
        md5_counts[md5_mod_bucket(s, k)] += 1
        hrw_counts[hrw_best_bucket(s, urls)] += 1

    md5_list = [md5_counts[i] for i in range(k)]
    hrw_list = [hrw_counts[i] for i in range(k)]

    print(f"scans_dir: {scans_dir}")
    print(f"num_scenes: {len(scene_ids)}")
    print(f"urls(k): {k}")
    for i, u in enumerate(urls):
        print(f"  [{i}] {u}")

    summarize_counts("MD5(scene_id) % k", md5_list)
    summarize_counts("HRW argmax md5(scene_id|url)", hrw_list)


if __name__ == "__main__":
    main()
