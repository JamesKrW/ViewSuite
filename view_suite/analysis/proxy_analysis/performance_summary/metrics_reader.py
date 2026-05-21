"""
Utilities for reading metrics.json files and computing success rates.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SAMPLE_ID_RE = re.compile(r"(scene\d+_\d+)_sample_(\d+)")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_all_metrics(task_dir: Path) -> List[Dict[str, Any]]:
    """Read all metrics.json files under a task directory."""
    files = sorted(task_dir.rglob("metrics.json"))
    results = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            d["_path"] = str(f)
            results.append(d)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Basic success rate
# ---------------------------------------------------------------------------

def compute_success_rate(metrics_list: List[Dict[str, Any]]) -> float:
    """Compute success rate from metrics list."""
    if not metrics_list:
        return 0.0
    successes = sum(1 for m in metrics_list if m.get("success", False))
    return successes / len(metrics_list)


# ---------------------------------------------------------------------------
# Detailed success (multi-threshold)
# ---------------------------------------------------------------------------

def _get_pos_ang(m: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract position and angle errors from metrics."""
    # Try top-level first
    pos = m.get("pos_err_m")
    ang = m.get("ang_err_deg")
    if pos is not None and ang is not None:
        return float(pos), float(ang)
    # Fallback to last info
    infos = m.get("infos", [])
    if infos and isinstance(infos[-1], dict):
        last = infos[-1]
        pos = last.get("pos_err_m")
        ang = last.get("ang_err_deg")
        if pos is not None and ang is not None:
            return float(pos), float(ang)
    return None, None


def compute_detailed_success_rates(
    metrics_list: List[Dict[str, Any]],
    thresholds: List[Tuple[float, float]],
    data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> Dict[Tuple[float, float], float]:
    """
    Compute success rate for each (pos_threshold, ang_threshold) pair.
    """
    counts = {t: [0, 0] for t in thresholds}  # [present, success]

    for m in metrics_list:
        pos, ang = _get_pos_ang(m)
        if pos is None or ang is None:
            continue
        for (pt, at) in thresholds:
            counts[(pt, at)][0] += 1
            if pos <= pt + 1e-9 and ang <= at + 1e-9:
                counts[(pt, at)][1] += 1

    return {
        t: (counts[t][1] / counts[t][0] if counts[t][0] > 0 else 0.0)
        for t in thresholds
    }


# ---------------------------------------------------------------------------
# Adaptive success
# ---------------------------------------------------------------------------

def _get_gt_action_len(
    m: Dict[str, Any],
    data_path: Optional[Path],
    jsonl_index: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Get ground truth action length from metrics, JSONL index, or meta.json."""
    # Try top-level or info0
    for source in [m, m.get("infos", [{}])[0] if m.get("infos") else {}]:
        gt_len = source.get("gt_action_len")
        if gt_len is not None:
            return int(gt_len)
        gt_action = source.get("gt_action") or source.get("gt_action_seq")
        if isinstance(gt_action, list):
            return len(gt_action)

    # Try JSONL index
    sample_id = _find_sample_id(m)
    if sample_id and jsonl_index and sample_id in jsonl_index:
        gt_seq = jsonl_index[sample_id].get("gt_action_seq")
        if isinstance(gt_seq, list):
            return len(gt_seq)

    # Try meta.json
    if sample_id and data_path:
        return _get_gt_len_from_meta(sample_id, data_path)

    return None


def _find_sample_id(m: Dict[str, Any]) -> Optional[str]:
    """Extract sample_id from metrics dict."""
    sid = m.get("sample_id")
    if sid:
        return str(sid)
    infos = m.get("infos", [])
    for info in infos:
        if isinstance(info, dict) and info.get("sample_id"):
            return str(info["sample_id"])
    return None


def _get_gt_len_from_meta(sample_id: str, data_path: Path) -> Optional[int]:
    """Get gt action length from meta.json."""
    match = _SAMPLE_ID_RE.match(sample_id)
    if not match:
        return None
    scene_id = match.group(1)
    sample_idx = match.group(2)
    meta_path = data_path / scene_id / f"sample_{sample_idx}" / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if "seq_len" in meta:
            return int(meta["seq_len"])
        for opt in meta.get("options", []):
            if opt.get("is_gt", False):
                return len(opt.get("action_seq", []))
    except Exception:
        pass
    return None


def _build_jsonl_index(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    """Build sample_id -> {gt_action_seq, ...} index from JSONL."""
    idx = {}
    if not jsonl_path or not jsonl_path.exists():
        return idx
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            sid = item.get("sample_id")
            if not sid:
                continue
            entry = {}
            action_seq = item.get("gt_action_seq")
            if isinstance(action_seq, list):
                entry["gt_action_seq"] = action_seq
            # Forward/inverse dynamics store action seq in meta
            meta = item.get("meta", {})
            if isinstance(meta, dict):
                meta_seq = meta.get("gt_action_seq_letters") or meta.get("gt_action_seq_names")
                if isinstance(meta_seq, list) and "gt_action_seq" not in entry:
                    entry["gt_action_seq"] = meta_seq
            gt_answer = item.get("gt_answer")
            if gt_answer:
                entry["gt_answer"] = gt_answer
            if entry:
                idx[str(sid)] = entry
    return idx


def compute_adaptive_success_rate(
    metrics_list: List[Dict[str, Any]],
    tol_per_action_len: str,
    data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> float:
    """Compute adaptive success rate based on action-length-dependent thresholds."""
    try:
        from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import (
            resolve_thresholds_per_action_len,
        )
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from view_suite.envs.scannet_proxy_task.utils.gym_proxy_tool_utils import (
            resolve_thresholds_per_action_len,
        )

    jsonl_index = _build_jsonl_index(jsonl_path) if jsonl_path else {}
    present = 0
    success = 0

    for m in metrics_list:
        pos, ang = _get_pos_ang(m)
        gt_len = _get_gt_action_len(m, data_path, jsonl_index)
        if pos is None or ang is None or gt_len is None:
            continue
        try:
            pt, at = resolve_thresholds_per_action_len(tol_per_action_len, gt_len)
        except Exception:
            continue
        present += 1
        if pos <= pt + 1e-9 and ang <= at + 1e-9:
            success += 1

    return success / present if present > 0 else 0.0


# ---------------------------------------------------------------------------
# Success by action length
# ---------------------------------------------------------------------------

def _get_interval_label(length: int, intervals: List[int]) -> str:
    """Get interval label for a given action length."""
    for i, threshold in enumerate(intervals):
        if length <= threshold:
            lower = intervals[i - 1] + 1 if i > 0 else 0
            return f"[{lower},{threshold}]"
    lower = intervals[-1] + 1
    return f"[{lower},+inf]"


def compute_success_by_action_len(
    metrics_list: List[Dict[str, Any]],
    intervals: List[int],
    data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute success rate grouped by action sequence length intervals.

    Returns dict: interval_label -> {total, success, success_rate}
    """
    jsonl_index = _build_jsonl_index(jsonl_path) if jsonl_path else {}
    interval_stats = defaultdict(lambda: {"total": 0, "success": 0})

    for m in metrics_list:
        gt_len = _get_gt_action_len(m, data_path, jsonl_index)
        if gt_len is None:
            continue
        label = _get_interval_label(gt_len, intervals)
        interval_stats[label]["total"] += 1
        if m.get("success", False):
            interval_stats[label]["success"] += 1

    results = {}
    for label in sorted(interval_stats.keys(), key=lambda x: int(x.split(",")[0].strip("["))):
        stats = interval_stats[label]
        total = stats["total"]
        success = stats["success"]
        results[label] = {
            "total": total,
            "success": success,
            "success_rate": success / total if total > 0 else 0.0,
        }

    # Overall
    total_all = sum(s["total"] for s in interval_stats.values())
    success_all = sum(s["success"] for s in interval_stats.values())
    results["_overall"] = {
        "total": total_all,
        "success": success_all,
        "success_rate": success_all / total_all if total_all > 0 else 0.0,
    }

    return results
