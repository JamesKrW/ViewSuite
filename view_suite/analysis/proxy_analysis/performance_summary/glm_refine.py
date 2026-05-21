"""
GLM format-error refinement.

GLM-4.6V often outputs bare letters (e.g., "A", "B") or wraps them in
<|begin_of_box|>X<|end_of_box|> without the expected <action>answer(X)</action>
format. The evaluation environment fails to parse these, resulting in
parsed_answer="" and format_reward=0.

This script:
1. Copies the original model rollout directory.
2. Re-parses each forward/inverse dynamics transcript to extract the answer.
3. Looks up the correct answer from the JSONL ground truth.
4. Updates metrics.json with corrected success, parsed_answer, rewards.
5. Regenerates summary.json.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Regex patterns to extract answer from various GLM formats
_ANSWER_PATTERNS = [
    # <action>answer(X)</action>
    re.compile(r"<action>\s*answer\(([A-D])\)\s*</action>"),
    # <|begin_of_box|><action>answer(X)</action><|end_of_box|>
    re.compile(r"answer\(([A-D])\)"),
    # <|begin_of_box|>answer(X)<|end_of_box|>
    re.compile(r"<\|begin_of_box\|>\s*answer\(([A-D])\)\s*<\|end_of_box\|>"),
    # <|begin_of_box|>X<|end_of_box|>
    re.compile(r"<\|begin_of_box\|>\s*([A-D])\s*<\|end_of_box\|>"),
    # Bare letter (last resort)
    re.compile(r"^([A-D])$"),
]


def _extract_answer(raw_response: str) -> Optional[str]:
    """Extract answer letter from raw model response."""
    raw = raw_response.strip()
    for pattern in _ANSWER_PATTERNS:
        m = pattern.search(raw)
        if m:
            return m.group(1)
    return None


def _build_jsonl_gt_index(jsonl_path: Path) -> Dict[int, str]:
    """Build jsonl_idx -> gt_answer mapping."""
    idx = {}
    if not jsonl_path.exists():
        return idx
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                gt = item.get("gt_answer")
                if gt:
                    idx[i] = str(gt).strip()
            except Exception:
                continue
    return idx


def _build_sample_gt_index(jsonl_path: Path) -> Dict[str, str]:
    """Build sample_id -> gt_answer mapping."""
    idx = {}
    if not jsonl_path.exists():
        return idx
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sid = item.get("sample_id")
                gt = item.get("gt_answer")
                if sid and gt:
                    idx[str(sid)] = str(gt).strip()
            except Exception:
                continue
    return idx


def refine_model(
    rollouts_dir: Path,
    model: str,
    data_path: Path,
) -> None:
    """
    Copy model dir and fix format errors in forward/inverse dynamics.
    """
    src = rollouts_dir / model
    dst = rollouts_dir / f"{model}_refined"

    if not src.exists():
        print(f"ERROR: Source directory not found: {src}")
        return

    # Copy directory
    if dst.exists():
        print(f"Destination already exists, removing: {dst}")
        shutil.rmtree(dst)

    print(f"Copying {src} -> {dst} ...")
    shutil.copytree(src, dst)
    print(f"Copy complete.")

    # Process forward and inverse dynamics
    jsonl_map = {
        "tag_path_to_view": data_path / "path_to_view_test_filter.jsonl",
        "tag_view_to_path": data_path / "view_to_path_test_filter.jsonl",
    }

    for task, jsonl_path in jsonl_map.items():
        task_dir = dst / task
        if not task_dir.is_dir():
            print(f"  {task}: MISSING, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Refining: {task}")
        print(f"{'='*60}")

        # Build GT index
        gt_by_idx = _build_jsonl_gt_index(jsonl_path)
        gt_by_sample = _build_sample_gt_index(jsonl_path)
        print(f"  GT index: {len(gt_by_idx)} entries (by idx), {len(gt_by_sample)} (by sample)")

        # Process each rollout
        metrics_files = sorted(task_dir.rglob("metrics.json"))
        total = len(metrics_files)
        fixed = 0
        already_ok = 0
        no_gt = 0
        no_answer = 0

        for mf in metrics_files:
            try:
                with open(mf) as f:
                    metrics = json.load(f)
            except Exception:
                continue

            infos = metrics.get("infos", [])
            if len(infos) < 2:
                continue

            last = infos[-1]
            info0 = infos[0]

            # Check if already has a parsed answer
            existing_parsed = last.get("parsed_answer", "")
            if existing_parsed:
                already_ok += 1
                continue

            # Get raw response
            raw = last.get("raw_response", "")
            if not raw:
                # Try reading from transcript
                transcript = mf.parent / "transcript.txt"
                if transcript.exists():
                    with open(transcript) as tf:
                        content = tf.read()
                    for line in content.split("\n"):
                        if line.startswith("ASSISTANT: ") or line.startswith("A: "):
                            raw = line.split(": ", 1)[1] if ": " in line else ""
                            break

            if not raw:
                no_answer += 1
                continue

            # Extract answer
            answer = _extract_answer(raw)
            if not answer:
                no_answer += 1
                continue

            # Get ground truth
            jsonl_idx = info0.get("jsonl_idx")
            sample_id = info0.get("sample_id")
            gt_answer = None
            if jsonl_idx is not None and jsonl_idx in gt_by_idx:
                gt_answer = gt_by_idx[jsonl_idx]
            elif sample_id and sample_id in gt_by_sample:
                gt_answer = gt_by_sample[sample_id]

            if not gt_answer:
                no_gt += 1
                continue

            # Determine correctness
            is_correct = (answer.upper() == gt_answer.upper())

            # Update last info
            last["parsed_answer"] = answer
            last["answer_correct"] = is_correct
            last["format_reward"] = 0.1  # fixed format
            last["answer_reward"] = 1.0 if is_correct else 0.0
            last["total_reward"] = 0.1 + (1.0 if is_correct else 0.0)
            last["success"] = is_correct
            last["_refined"] = True

            # Update top-level metrics
            metrics["success"] = is_correct
            metrics["cumulative_reward"] = last["total_reward"]

            # Write back
            with open(mf, "w") as f:
                json.dump(metrics, f, indent=2)

            fixed += 1

        print(f"  Total: {total}")
        print(f"  Already OK (had parsed_answer): {already_ok}")
        print(f"  Fixed: {fixed}")
        print(f"  No GT found: {no_gt}")
        print(f"  No answer extractable: {no_answer}")

        # Regenerate summary.json
        _regenerate_summary(task_dir)

    print(f"\nDone. Refined rollouts at: {dst}")


def _regenerate_summary(task_dir: Path) -> None:
    """Regenerate summary.json for a task directory."""
    summary_path = task_dir / "summary.json"

    metrics_files = sorted(task_dir.rglob("metrics.json"))
    if not metrics_files:
        return

    n_episodes = len(metrics_files)
    successes = 0
    total_reward = 0.0
    total_turns = 0
    episodes = []

    for mf in metrics_files:
        try:
            with open(mf) as f:
                m = json.load(f)
        except Exception:
            continue

        success = m.get("success", False)
        if success:
            successes += 1
        reward = m.get("cumulative_reward", 0)
        total_reward += reward
        turns = m.get("num_turns", 1)
        total_turns += turns

    # Update existing summary or create new one
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["n_episodes"] = n_episodes
    summary["success_rate"] = successes / n_episodes if n_episodes > 0 else 0.0
    summary["avg_cumulative_reward"] = total_reward / n_episodes if n_episodes > 0 else 0.0
    summary["avg_turns"] = total_turns / n_episodes if n_episodes > 0 else 0.0
    summary["_refined"] = True

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Updated summary: success_rate={summary['success_rate']:.4f} ({successes}/{n_episodes})")
