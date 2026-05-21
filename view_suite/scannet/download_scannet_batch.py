# scannet_utils.py
# -*- coding: utf-8 -*-
"""
Parallel ScanNet downloader utilities with tqdm progress (thread/process safe).

Exposes a single function:
    download_scannet_batch(scannet_download_script_path, out_dir, scene_ids,
                           file_types=(".sens","_vh_clean.ply",".txt"),
                           timeout_s=900, num_workers=6, retries=2, retry_delay_s=3.0,
                           executor="thread", verbose=False, tqdm_unit="file",
                           progress=True, progress_json_path="")

Features:
- Skips files that already exist, and counts them into progress on start.
- Lazy directory creation: only create scans/<scene_id>/ when actually downloading that file.
- Parallel downloads via ThreadPoolExecutor or ProcessPoolExecutor (with safe fallback).
- Per-item retries with a small backoff.
- Clean tqdm progress bar in main process; optional verbose logs in workers.
- Tracks both scene-level "finished" (all items done) and "ready" (all items present on disk).
- Writes a small progress JSON snapshot for external monitoring (e.g., tmux status bar).
"""

from __future__ import annotations

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Sequence, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm


def _expected_path(out_root: Path, scene_id: str, file_type: str) -> Path:
    """
    Compute the expected output path for a given scene/file_type.
    NOTE: Do NOT mkdir here; we lazily create the parent when we actually download.
    """
    return (out_root / "scans" / scene_id) / f"{scene_id}{file_type}"


def _run_download_once(
    scannet_download_script_path: Path,
    out_root: Path,
    scene_id: str,
    file_type: str,
    timeout_s: int,
    tos_answer: str = "y",
    tos_repeat: int = 8,
) -> Tuple[int, str]:
    """
    Run the official download-scannet.py once for (scene_id, file_type).
    Returns (rc, tail_msg).
    """
    cmd = [
        sys.executable, str(scannet_download_script_path),
        "-o", str(out_root),
        "--id", scene_id,
        "--type", file_type,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Feed multiple answers to satisfy scripts that call input() more than once.
        # e.g., "Do you accept? [y/n]" or multiple confirmations.
        # We send "y\n" * tos_repeat by default.
        feed = ((tos_answer or "y").strip() + "\n") * max(1, tos_repeat)
        out_text, err_text = proc.communicate(input=feed, timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return 124, "timeout"
    # Keep a short tail (stderr preferred; fallback to stdout)
    tail_src = (err_text if err_text else out_text or "").strip()
    tail_msg = tail_src[-300:] if tail_src else ""
    return rc, tail_msg


def _download_task(
    scannet_download_script_path: Path,
    out_root: Path,
    scene_id: str,
    file_type: str,
    timeout_s: int,
    retries: int,
    retry_delay_s: float,
    verbose: bool = False,
    tos_answer: str = "y",
    tos_repeat: int = 8,
) -> Tuple[str, str, str]:
    """
    Single download task with retry.
    Returns (scene_id, file_type, status) where status in {"ok","skipped","fail"}.
    """
    expected = _expected_path(out_root, scene_id, file_type)
    if expected.exists():
        if verbose:
            print(f"[SKIP] already exists: {expected}")
        return scene_id, file_type, "skipped"

    # Lazily create the target folder only when we are about to download this item.
    expected.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        attempt += 1
        if verbose:
            print(f"[INFO] downloading {scene_id} {file_type} (attempt {attempt}) ...")
        rc, tail = _run_download_once(scannet_download_script_path, out_root, scene_id, file_type, timeout_s, tos_answer, tos_repeat)
        if rc == 0:
            if verbose:
                print(f"[OK] {scene_id} {file_type}")
            return scene_id, file_type, "ok"
        if attempt > retries + 1:  # first try + 'retries' re-tries
            if verbose:
                print(f"[ERR] {scene_id} {file_type} failed rc={rc}. tail={tail}")
            return scene_id, file_type, "fail"
        if verbose:
            print(f"[RETRY] {scene_id} {file_type} rc={rc}. tail={tail} -> retrying in {retry_delay_s}s")
        time.sleep(retry_delay_s)


def download_scannet_batch(
    scannet_download_script_path: str,
    out_dir: str,
    scene_ids: Sequence[str],
    file_types: Sequence[str] = (".sens", "_vh_clean.ply", ".txt"),
    timeout_s: int = 900,
    num_workers: int = 6,
    retries: int = 2,
    retry_delay_s: float = 3.0,
    executor: str = "thread",  # "thread" | "process"
    verbose: bool = False,     # worker logs; keep False to avoid messing tqdm
    tqdm_unit: str = "file",
    progress: bool = True,
    progress_json_path: str = "",
    tos_answer: str = "y",
    tos_repeat: int = 8,
) -> Dict[str, int]:
    """
    Download selected file_types for given scene_ids using the official download-scannet.py.
    - Parallel execution (num_workers threads/processes).
    - Skips files that already exist; counts them into progress at start.
    - Retries failed downloads a limited number of times.

    Returns:
        dict summary with counts:
        {
          "scenes", "attempted", "ok", "fail", "skipped",
          "scenes_finished", "scenes_ready"
        }
    """
    dl_py = Path(scannet_download_script_path)
    out_root = Path(out_dir)
    # Only create the top-level output dir; no per-scene pre-creation.
    out_root.mkdir(parents=True, exist_ok=True)

    # Build tasks for items that don't already exist (no directory creation here).
    tasks: List[Tuple[str, str]] = []
    skipped_files_initial = 0

    # Per-scene counters
    per_scene_total: Dict[str, int] = {sid: len(file_types) for sid in scene_ids}
    per_scene_done: Dict[str, int] = {sid: 0 for sid in scene_ids}     # terminal states: ok/fail/skip
    per_scene_present: Dict[str, int] = {sid: 0 for sid in scene_ids}  # files present on disk (pre-exist or ok/skip)

    for sid in scene_ids:
        for ft in file_types:
            expected = _expected_path(out_root, sid, ft)
            if expected.exists():
                if verbose:
                    print(f"[SKIP] already exists: {expected}")
                skipped_files_initial += 1
                per_scene_done[sid] += 1
                per_scene_present[sid] += 1
            else:
                tasks.append((sid, ft))

    scenes_total = len(scene_ids)
    items_total = len(scene_ids) * len(file_types)

    # Scenes state at start
    scenes_finished = sum(1 for sid in scene_ids if per_scene_done[sid] >= per_scene_total[sid])
    scenes_ready = sum(1 for sid in scene_ids if per_scene_present[sid] >= per_scene_total[sid])

    # Global file-level counters at start
    ok = 0
    fail = 0
    skipped = skipped_files_initial
    items_done = skipped_files_initial

    def dump_progress_json():
        if not progress_json_path:
            return
        payload = {
            "scenes_total": scenes_total,
            "scenes_finished": scenes_finished,
            "scenes_ready": scenes_ready,
            "files_total": items_total,
            "files_done": items_done,
            "ok": ok,
            "fail": fail,
            "skip": skipped,
            "timestamp": time.time(),
        }
        try:
            Path(progress_json_path).write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    # Prepare tqdm bar in main process
    pbar = None
    if progress:
        desc = f"scenes {scenes_finished}/{scenes_total} | ready {scenes_ready}/{scenes_total}"
        pbar = tqdm(
            total=items_total,
            initial=items_done,
            unit=tqdm_unit,
            dynamic_ncols=True,
            desc=desc,
            disable=not sys.stdout.isatty(),  # still prints summary if not TTY
        )
        if pbar is not None:
            pbar.set_postfix(ok=ok, fail=fail, skip=skipped)
        dump_progress_json()

    if not tasks:
        summary = {
            "scenes": scenes_total,
            "attempted": 0,
            "ok": 0,
            "fail": 0,
            "skipped": skipped,
            "scenes_finished": scenes_finished,
            "scenes_ready": scenes_ready,
        }
        if pbar is not None:
            pbar.close()
        print("[SUMMARY]", summary)
        dump_progress_json()
        return summary

    print(f"[INFO] starting downloads: {len(tasks)} items with {max(1, num_workers)} {executor}s")

    # Try the requested executor; on failure (e.g., missing __main__ guard in caller), fallback to threads.
    ExecutorCls = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
    tried_process = (executor == "process")
    try:
        context_mgr = ExecutorCls(max_workers=max(1, num_workers))
    except Exception as e:
        if tried_process:
            print(f"[WARN] process executor unavailable ({e!r}); falling back to thread executor.")
            context_mgr = ThreadPoolExecutor(max_workers=max(1, num_workers))
            tried_process = False
        else:
            raise

    with context_mgr as ex:
        futures = [
            ex.submit(
                _download_task,
                dl_py, out_root, sid, ft,
                timeout_s, retries, retry_delay_s, verbose, tos_answer, tos_repeat
            )
            for (sid, ft) in tasks
        ]
        for fut in as_completed(futures):
            sid, ft, status = fut.result()  # type: ignore
            # Update file-level counters
            items_done += 1
            if status == "ok":
                ok += 1
                per_scene_done[sid] += 1
                per_scene_present[sid] += 1
            elif status == "fail":
                fail += 1
                per_scene_done[sid] += 1
            else:  # "skipped" (appeared between build and run)
                skipped += 1
                per_scene_done[sid] += 1
                per_scene_present[sid] += 1

            # Update scene-level completion/ready counts on threshold crossings
            if per_scene_done[sid] == per_scene_total[sid]:  # finished (ok/fail/skip all resolved)
                scenes_finished += 1
            if per_scene_present[sid] == per_scene_total[sid]:  # ready on disk
                scenes_ready += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"scenes {scenes_finished}/{scenes_total} | ready {scenes_ready}/{scenes_total}")
                pbar.set_postfix(ok=ok, fail=fail, skip=skipped)
            dump_progress_json()

    summary = {
        "scenes": scenes_total,
        "attempted": len(tasks),
        "ok": ok,
        "fail": fail,
        "skipped": skipped,
        "scenes_finished": scenes_finished,
        "scenes_ready": scenes_ready,
    }
    if pbar is not None:
        pbar.close()
    print("[SUMMARY]", summary)
    dump_progress_json()
    return summary


# Optional CLI (only runs when invoked directly)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel ScanNet downloader with tqdm.")
    parser.add_argument("--dl", required=True, help="Path to official download-scannet.py")
    parser.add_argument("--out", required=True, help="Output root dir")
    parser.add_argument("--scenes", required=True, nargs="+", help="Scene IDs (e.g., scene0000_00 scene0001_00)")
    parser.add_argument("--types", nargs="+", default=[".sens", "_vh_clean.ply", ".txt"], help="File types to fetch")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=3.0)
    parser.add_argument("--executor", choices=["thread", "process"], default="thread")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--progress_json", default="")
    args = parser.parse_args()

    download_scannet_batch(
        scannet_download_script_path=args.dl,
        out_dir=args.out,
        scene_ids=args.scenes,
        file_types=tuple(args.types),
        timeout_s=args.timeout,
        num_workers=args.workers,
        retries=args.retries,
        retry_delay_s=args.retry_delay,
        executor=args.executor,
        verbose=args.verbose,
        progress=(not args.no_progress),
        progress_json_path=args.progress_json,
    )
