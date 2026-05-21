"""
Generate paper-quality coverage analysis figures for Section 5.3.

Main figure: 2 subplots (coverage ratio + target intersection ratio) with key models.
Appendix figure: same layout with ALL models.

Usage:
    source /opt/miniforge3/etc/profile.d/conda.sh && conda activate viewsuite
    python scripts/plot_coverage_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/root/projects/viewsuite/rollouts_pointcloud_coverage")
OUT_MAIN = Path("/root/projects/viewsuite_paper/69a8d3d467b91730ac6c6628/sections/5_analysis/figures")
OUT_APPENDIX = Path("/root/projects/viewsuite_paper/69a8d3d467b91730ac6c6628/appendix_sections/e_analysis/figures")

# ── Model definitions ──────────────────────────────────────────────────
# (dir_name, display_label, color, linestyle, linewidth, marker, zorder)
MAIN_MODELS = [
    ("qwen_25_vl_7b_trained", "SERS (Ours)",         "#D946EF", "-",  3.0, "o", 20),
    ("qwen_25_vl_7b",         "Qwen2.5-VL-7B (Base)", "#9CA3AF", "--", 2.0, "s", 10),
    ("gpt_5_4_pro",           "GPT-5.4 Pro",          "#2563EB", "-",  1.8, "^", 8),
    ("gemini_3_1_pro",        "Gemini 3.1 Pro",       "#EF4444", "-",  1.8, "v", 8),
    ("qwen3_5_397b",          "Qwen3.5-397B",         "#F59E0B", "-",  1.8, "D", 8),
    ("claude_opus_4_6",       "Claude Opus 4.6",      "#10B981", "-",  1.8, "P", 8),
]

ALL_MODELS = [
    ("qwen_25_vl_7b_trained", "SERS (Ours)",         "#D946EF", "-",  3.0, "o", 20),
    ("qwen_25_vl_7b",         "Qwen2.5-VL-7B (Base)", "#9CA3AF", "--", 2.0, "s", 10),
    ("gpt_5_4_pro",           "GPT-5.4 Pro",          "#2563EB", "-",  1.5, "^", 5),
    ("gpt_5_4",               "GPT-5.4",              "#3B82F6", "-",  1.5, "v", 5),
    ("gpt_5_1",               "GPT-5.1",              "#60A5FA", "-",  1.5, "<", 5),
    ("gemini_3_1_pro",        "Gemini 3.1 Pro",       "#EF4444", "-",  1.5, ">", 5),
    ("gemini_3_pro",          "Gemini 3 Pro",         "#F87171", "-",  1.5, "d", 5),
    ("claude_opus_4_6",       "Claude Opus 4.6",      "#10B981", "-",  1.5, "P", 5),
    ("grok_4_20_beta",        "Grok 4.20 Beta",       "#F97316", "-",  1.5, "X", 5),
    ("qwen3_5_397b",          "Qwen3.5-397B",         "#F59E0B", "-",  1.5, "D", 5),
    ("qwen2_5_vl_72b",        "Qwen2.5-VL-72B",      "#A855F7", "-",  1.5, "h", 5),
    ("qwen3_vl_32b",          "Qwen3-VL-32B",         "#8B5CF6", "-",  1.5, "p", 5),
    ("glm_4_6v",              "GLM-4.6V",             "#06B6D4", "-",  1.5, "*", 5),
    ("glm_4_6v_refined",      "GLM-4.6V (Refined)",   "#0891B2", "--", 1.5, "*", 5),
    ("kimi_k2_5",             "Kimi K2.5",            "#84CC16", "-",  1.5, "H", 5),
]


def load_summary(dir_name: str) -> dict:
    with open(DATA_ROOT / dir_name / "summary.json") as f:
        return json.load(f)


def extract_series(stats_by_turn: dict):
    turns = sorted(int(k) for k in stats_by_turn.keys())
    means = [stats_by_turn[str(t)]["mean"] for t in turns]
    stds = [stats_by_turn[str(t)]["std"] for t in turns]
    counts = [stats_by_turn[str(t)]["count"] for t in turns]
    # Filter sparse turns
    max_count = max(counts)
    valid = [j for j, c in enumerate(counts) if c >= max_count * 0.01]
    return (
        [turns[j] for j in valid],
        [means[j] for j in valid],
        [stds[j] for j in valid],
    )


def plot_two_panel(models, out_dir: Path, filename: str, figsize=(13, 5.2),
                   legend_fontsize=20, tick_fontsize=22, label_fontsize=24):
    """Plot coverage ratio and target intersection ratio side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    legend_handles = []
    legend_labels = []

    for dir_name, label, color, ls, lw, marker, zorder in models:
        summary = load_summary(dir_name)
        n = summary["num_trajectories"]

        # (a) Coverage ratio
        turns, means, stds = extract_series(summary["coverage_ratio"])
        ax1.plot(turns, means, color=color, linestyle=ls, linewidth=lw,
                 marker=marker, markersize=5, zorder=zorder)
        ax1.fill_between(turns,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.10, color=color, zorder=zorder - 1)

        # (b) Target intersection ratio
        if "target_intersection_ratio" in summary:
            turns_t, means_t, stds_t = extract_series(summary["target_intersection_ratio"])
            ax2.plot(turns_t, means_t, color=color, linestyle=ls, linewidth=lw,
                     marker=marker, markersize=5, zorder=zorder)
            ax2.fill_between(turns_t,
                             np.array(means_t) - np.array(stds_t),
                             np.array(means_t) + np.array(stds_t),
                             alpha=0.10, color=color, zorder=zorder - 1)

        # Legend entry
        handle = Line2D([0], [0], color=color, linestyle=ls, linewidth=lw,
                        marker=marker, markersize=5)
        legend_handles.append(handle)
        legend_labels.append(label)

    # Formatting
    for ax, ylabel, title_letter, title_text in [
        (ax1, "Scene Coverage Ratio", "(a)", "Scene Coverage"),
        (ax2, "Target Intersection Ratio", "(b)", "Target Intersection"),
    ]:
        ax.set_xlabel("Turn", fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_title(f"{title_letter} {title_text}", fontsize=28, pad=12)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.3, 10.3)
        # Integer x-ticks
        ax.set_xticks(range(0, 11))

    ax1.set_ylim(0.04, 0.24)
    ax2.set_ylim(0.0, 0.70)

    # Shared legend at bottom
    fig.legend(legend_handles, legend_labels,
               loc="lower center", ncol=min(len(models), 6),
               fontsize=legend_fontsize, frameon=True, framealpha=0.9,
               edgecolor="#E5E7EB", borderpad=0.4, columnspacing=1.0,
               handletextpad=0.5, handlelength=2.0)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = out_dir / f"{filename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Main figure (Section 5.3) ===")
    plot_two_panel(MAIN_MODELS, OUT_MAIN, "coverage_analysis",
                   figsize=(20, 7))

    print("\n=== Appendix figure (all models) ===")
    plot_two_panel(ALL_MODELS, OUT_APPENDIX, "coverage_analysis_all",
                   figsize=(22, 8))

    print("\nDone.")
