"""
Attention heatmap analysis for Qwen2.5-VL trajectories.

Extracts per-turn attention from specified model layer(s), maps vision-token
attention back to image pixels, and overlays heatmaps on the original
observation images.

Usage:
    # Single trajectory, last layer
    python -m view_suite.analysis.scannet_attn.main run \
        --traj_dir /path/to/trajectory \
        --model_path /path/to/qwen25vl

    # Multiple layers
    python -m view_suite.analysis.scannet_attn.main run \
        --traj_dir /path/to/trajectory \
        --model_path /path/to/qwen25vl \
        --layer_indices '[-1, 14, 0]'

    # Batch
    python -m view_suite.analysis.scannet_attn.main run_batch \
        --rollout_dir /path/to/rollout_dir \
        --model_path /path/to/qwen25vl \
        --max_trajs 10
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Union

import fire

from view_suite.analysis.scannet_attn.model_manager import ModelManager
from view_suite.analysis.scannet_attn.attention_extractor import (
    extract_trajectory_attention,
)
from view_suite.analysis.scannet_attn.heatmap_visualizer import save_heatmap_image


def _parse_layer_indices(layer_indices) -> List[int]:
    """Parse layer_indices from CLI: int, list, or JSON string."""
    if isinstance(layer_indices, int):
        return [layer_indices]
    if isinstance(layer_indices, (list, tuple)):
        return [int(x) for x in layer_indices]
    if isinstance(layer_indices, str):
        parsed = json.loads(layer_indices)
        if isinstance(parsed, int):
            return [parsed]
        return [int(x) for x in parsed]
    return [int(layer_indices)]


class AttentionAnalyzer:
    """CLI interface for attention heatmap analysis."""

    def run(
        self,
        traj_dir: str,
        model_path: str,
        layer_indices: Union[int, str] = -1,
        alpha: float = 0.4,
        colormap: str = "jet",
        output_subdir: str = "attention_map",
        device: str = "cuda:0",
    ) -> None:
        """
        Analyze a single trajectory.

        Args:
            traj_dir: Path to trajectory directory.
            model_path: Path to Qwen2.5-VL model checkpoint.
            layer_indices: Layer(s) to extract attention from.
                           Int (-1 = last) or JSON list e.g. '[-1, 14, 0]'.
            alpha: Heatmap overlay transparency (0 = no heatmap, 1 = full).
            colormap: Matplotlib colormap name (jet, viridis, hot, etc.).
            output_subdir: Subdirectory name for output within traj_dir.
            device: CUDA device.
        """
        indices = _parse_layer_indices(layer_indices)
        manager = ModelManager(
            model_path, layer_indices=indices, device=device
        )
        manager.load()

        _process_one_trajectory(
            traj_dir, manager, alpha, colormap, output_subdir
        )

    def run_batch(
        self,
        rollout_dir: str,
        model_path: str,
        layer_indices: Union[int, str] = -1,
        alpha: float = 0.4,
        colormap: str = "jet",
        output_subdir: str = "attention_map",
        device: str = "cuda:0",
        max_trajs: int = 0,
    ) -> None:
        """
        Analyze multiple trajectories in a rollout directory.

        Args:
            rollout_dir: Directory containing trajectory subdirectories.
            model_path: Path to Qwen2.5-VL model checkpoint.
            layer_indices: Layer(s) to extract attention from.
            alpha: Heatmap overlay transparency.
            colormap: Matplotlib colormap name.
            output_subdir: Subdirectory name for output within each traj dir.
            device: CUDA device.
            max_trajs: Maximum trajectories to process (0 = all).
        """
        indices = _parse_layer_indices(layer_indices)
        manager = ModelManager(
            model_path, layer_indices=indices, device=device
        )
        manager.load()

        rollout_path = Path(rollout_dir)
        traj_dirs = sorted([
            d for d in rollout_path.iterdir()
            if d.is_dir() and (d / "messages.json").exists()
        ])

        if max_trajs > 0:
            traj_dirs = traj_dirs[:max_trajs]

        print(f"\nProcessing {len(traj_dirs)} trajectories from {rollout_dir}")
        t0 = time.time()

        for i, traj_path in enumerate(traj_dirs):
            print(f"\n[{i + 1}/{len(traj_dirs)}] {traj_path.name}")
            try:
                _process_one_trajectory(
                    str(traj_path), manager, alpha, colormap, output_subdir
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed = time.time() - t0
        print(f"\nDone. {len(traj_dirs)} trajectories in {elapsed:.1f}s")


def _process_one_trajectory(
    traj_dir: str,
    manager: ModelManager,
    alpha: float,
    colormap: str,
    output_subdir: str,
) -> None:
    """Process a single trajectory: extract attention -> save heatmaps.

    Output structure:
        <output_subdir>/layer_<L>/turn_<T>/<image_filename>.png
    Each turn folder contains heatmaps for all images visible at that turn.
    """
    traj_path = Path(traj_dir)
    output_base = traj_path / output_subdir
    output_base.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = extract_trajectory_attention(traj_dir, manager)

    images_dir = traj_path / "images"
    saved = 0

    for turn_result in result.turns:
        layer_idx = turn_result.layer_idx
        turn_idx = turn_result.turn_idx

        # attention_map/layer_XX/turn_XX/
        turn_dir = output_base / f"layer_{layer_idx}" / f"turn_{turn_idx:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        for img_fname, heatmap in turn_result.image_heatmaps.items():
            src_img = str(images_dir / img_fname)
            dst_img = str(turn_dir / img_fname)

            save_heatmap_image(
                src_img, heatmap, dst_img,
                alpha=alpha, colormap=colormap, add_colorbar=True,
            )
            saved += 1

    # Save metadata
    metadata = {
        "traj_dir": str(traj_dir),
        "model_path": manager.model_path,
        "layer_indices": manager.layer_indices,
        "num_heads": manager.num_heads,
        "alpha": alpha,
        "colormap": colormap,
        "num_turns": len(result.turns),
        "num_images": saved,
        "image_filenames": result.image_filenames,
    }
    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Saved {saved} heatmaps to {output_base} ({elapsed:.1f}s)")


if __name__ == "__main__":
    fire.Fire(AttentionAnalyzer)
