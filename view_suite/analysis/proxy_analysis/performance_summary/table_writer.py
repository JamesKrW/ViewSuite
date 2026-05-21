"""
Utilities for writing CSV and Markdown tables.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse display name mapping from main
_MODEL_DISPLAY = {
    "claude_opus_4_6": "Claude Opus 4.6",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_3_pro": "Gemini 3 Pro",
    "glm_4_6v": "GLM-4.6V",
    "glm_4_6v_refined": "GLM-4.6V (refined)",
    "gpt_5_1": "GPT-5.1",
    "gpt_5_4": "GPT-5.4",
    "gpt_5_4_pro": "GPT-5.4 Pro",
    "grok_4_20_beta": "Grok 4.20 Beta",
    "qwen2_5_vl_72b": "Qwen2.5-VL-72B",
    "qwen3_5_397b": "Qwen3.5-397B",
    "qwen3_vl_32b": "Qwen3-VL-32B",
    "qwen_25_vl_7b": "Qwen2.5-VL-7B",
    "qwen_25_vl_7b_trained": "Qwen2.5-VL-7B (trained)",
}


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _pct(x: float) -> str:
    return f"{x*100:.1f}"


def _threshold_key(m: float, d: float) -> str:
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}".replace(".", "p").replace("-", "neg")
    return f"success_{fmt(m)}m{fmt(d)}degree"


def _threshold_label(m: float, d: float) -> str:
    def fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:g}"
    return f"{fmt(m)}m/{fmt(d)}deg"


# ---------------------------------------------------------------------------
# Per-model CSV
# ---------------------------------------------------------------------------

def write_per_model_csv(
    result: Dict[str, Any],
    thresholds: List[Tuple[float, float]],
    path: Path,
) -> None:
    """Write a single model's performance to CSV."""
    rows = []
    rows.append(("Metric", "Value"))
    rows.append(("Model", result.get("display_name", result["model"])))

    for task_key, task_label in [
        ("tag_path_to_view_success", "Path2View Success"),
        ("tag_view_to_path_success", "View2Path Success"),
        ("tag_interactive_view_planning_success", "Interactive View Planning Success (default)"),
    ]:
        if task_key in result:
            rows.append((task_label, _fmt(result[task_key])))

    for m, d in thresholds:
        key = _threshold_key(m, d)
        if key in result:
            rows.append((f"AE {_threshold_label(m, d)}", _fmt(result[key])))

    if "adaptive_success" in result:
        rows.append(("AE Adaptive Success", _fmt(result["adaptive_success"])))
    if "ae_avg_success" in result:
        rows.append(("AE Avg Threshold Success", _fmt(result["ae_avg_success"])))
    if "overall_score" in result:
        rows.append(("Overall Score", _fmt(result["overall_score"])))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Combined CSV
# ---------------------------------------------------------------------------

def write_combined_csv(
    all_results: Dict[str, Dict[str, Any]],
    thresholds: List[Tuple[float, float]],
    path: Path,
) -> None:
    """Write combined performance table as CSV."""
    header = ["Model", "Forward Dyn.", "Inverse Dyn."]
    for m, d in thresholds:
        header.append(f"AE {_threshold_label(m, d)}")
    header.extend(["AE Adaptive", "AE Avg", "Overall Score"])

    rows = [header]
    # Sort by overall_score descending
    sorted_models = sorted(all_results.keys(), key=lambda k: all_results[k].get("overall_score", 0), reverse=True)

    for model in sorted_models:
        r = all_results[model]
        row = [r.get("display_name", model)]
        row.append(_pct(r.get("tag_path_to_view_success", 0)))
        row.append(_pct(r.get("tag_view_to_path_success", 0)))
        for m, d in thresholds:
            key = _threshold_key(m, d)
            row.append(_pct(r.get(key, 0)))
        row.append(_pct(r.get("adaptive_success", 0)))
        row.append(_pct(r.get("ae_avg_success", 0)))
        row.append(_pct(r.get("overall_score", 0)))
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Combined Markdown
# ---------------------------------------------------------------------------

def write_combined_md(
    all_results: Dict[str, Dict[str, Any]],
    thresholds: List[Tuple[float, float]],
    path: Path,
) -> None:
    """Write combined performance table as Markdown."""
    header = ["Model", "Forward Dyn.", "Inverse Dyn."]
    for m, d in thresholds:
        header.append(f"AE {_threshold_label(m, d)}")
    header.extend(["AE Adaptive", "AE Avg", "Overall Score"])

    sorted_models = sorted(all_results.keys(), key=lambda k: all_results[k].get("overall_score", 0), reverse=True)

    lines = []
    lines.append("# Performance Summary\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for model in sorted_models:
        r = all_results[model]
        row = [r.get("display_name", model)]
        row.append(_pct(r.get("tag_path_to_view_success", 0)))
        row.append(_pct(r.get("tag_view_to_path_success", 0)))
        for m, d in thresholds:
            key = _threshold_key(m, d)
            row.append(_pct(r.get(key, 0)))
        row.append(_pct(r.get("adaptive_success", 0)))
        row.append(_pct(r.get("ae_avg_success", 0)))
        row.append(_pct(r.get("overall_score", 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*All values are percentages (%)*")
    lines.append("")
    lines.append("**Thresholds:**")
    lines.append(f"- AE columns: position/angle tolerance for active exploration success")
    lines.append(f"- AE Adaptive: adaptive threshold based on action length")
    lines.append(f"- AE Avg: average across all threshold columns")
    lines.append(f"- Overall Score: average of Forward Dyn., Inverse Dyn., and AE Avg")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Action-length CSV
# ---------------------------------------------------------------------------

def write_action_len_csv(
    all_results: Dict[str, Dict[str, Any]],
    intervals: List[int],
    path: Path,
    task_key: str = "tag_interactive_view_planning",
) -> None:
    """Write success-by-action-length table as CSV."""
    # Build interval labels
    labels = []
    for i, threshold in enumerate(intervals):
        lower = intervals[i - 1] + 1 if i > 0 else 0
        labels.append(f"[{lower},{threshold}]")
    labels.append(f"[{intervals[-1] + 1},+inf]")

    header = ["Model"] + [f"{l} (rate)" for l in labels] + [f"{l} (n)" for l in labels] + ["Overall"]
    rows = [header]

    for model, data in sorted(all_results.items()):
        task_data = data.get(task_key, {})
        if not task_data:
            continue
        display = _MODEL_DISPLAY.get(model, model)
        row = [display]
        # Rates
        for label in labels:
            info = task_data.get(label, {})
            row.append(_pct(info.get("success_rate", 0)))
        # Counts
        for label in labels:
            info = task_data.get(label, {})
            row.append(str(info.get("total", 0)))
        # Overall
        overall = task_data.get("_overall", {})
        row.append(_pct(overall.get("success_rate", 0)))
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Action-length Markdown
# ---------------------------------------------------------------------------

def write_action_len_md(
    all_results: Dict[str, Dict[str, Any]],
    intervals: List[int],
    path: Path,
    task_key: str = "tag_interactive_view_planning",
    title: str = "Success Rate by Action Sequence Length",
) -> None:
    """Write success-by-action-length table as Markdown."""
    labels = []
    for i, threshold in enumerate(intervals):
        lower = intervals[i - 1] + 1 if i > 0 else 0
        labels.append(f"[{lower},{threshold}]")
    labels.append(f"[{intervals[-1] + 1},+inf]")

    header = ["Model"] + labels + ["Overall"]

    lines = []
    lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    sorted_models = sorted(all_results.keys())
    for model in sorted_models:
        data = all_results[model]
        task_data = data.get(task_key, {})
        if not task_data:
            continue
        display = _MODEL_DISPLAY.get(model, model)
        row = [display]
        for label in labels:
            info = task_data.get(label, {})
            total = info.get("total", 0)
            rate = info.get("success_rate", 0)
            row.append(f"{rate*100:.1f}% ({total})")
        overall = task_data.get("_overall", {})
        row.append(f"{overall.get('success_rate', 0)*100:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("*Format: success_rate% (count)*")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")
