"""
Extract per-turn attention maps from a Qwen2.5-VL trajectory replay.

Pipeline:
  1. Reconstruct conversation from messages.json + image files
  2. Tokenize with the Qwen2.5-VL processor
  3. Single forward pass (teacher-forcing) with output_attentions=True
  4. For each assistant turn, extract attention from response tokens → image tokens
  5. Aggregate across heads → per-vision-token scores → spatial heatmap
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from view_suite.analysis.scannet_attn.model_manager import ModelManager
from view_suite.analysis.scannet_attn.token_image_mapper import (
    ImageTokenSpan,
    ResponseTokenSpan,
    attention_to_heatmap,
    find_image_token_spans,
    find_response_token_spans,
)


@dataclass
class TurnAttentionResult:
    """Attention heatmaps produced by one assistant turn, for one layer."""

    turn_idx: int
    layer_idx: int
    image_heatmaps: Dict[str, np.ndarray] = field(default_factory=dict)
    # key = image filename (e.g. "turn_01_01.png"), value = (H, W) heatmap


@dataclass
class TrajectoryAttentionResult:
    """All attention results for one trajectory."""

    traj_dir: str
    turns: List[TurnAttentionResult] = field(default_factory=list)
    image_filenames: List[str] = field(default_factory=list)
    layer_indices: List[int] = field(default_factory=list)


def extract_trajectory_attention(
    traj_dir: str,
    manager: ModelManager,
    target_h: int = 512,
    target_w: int = 512,
) -> TrajectoryAttentionResult:
    """
    Run a single trajectory through the model and extract attention heatmaps.

    Args:
        traj_dir: path to trajectory directory (contains messages.json, images/)
        manager: loaded ModelManager
        target_h, target_w: output heatmap resolution

    Returns:
        TrajectoryAttentionResult with per-turn, per-layer, per-image heatmaps
    """
    traj_path = Path(traj_dir)

    # ---- 1. Reconstruct conversation ----
    messages, image_files = _rebuild_conversation(traj_path)

    # ---- 2. Tokenize ----
    inputs, input_ids_flat = _tokenize_conversation(
        messages, image_files, manager
    )

    # ---- 3. Forward pass ----
    attn_by_layer = _forward_with_attention(inputs, manager)
    # dict: layer_idx → (1, num_heads, seq_len, seq_len)

    # ---- 4. Find token spans ----
    image_spans = find_image_token_spans(
        input_ids_flat,
        inputs["image_grid_thw"],
        manager.spatial_merge_size,
    )
    response_spans = find_response_token_spans(
        input_ids_flat,
        manager.processor.tokenizer,
    )

    # ---- 5. Build turn → image mapping ----
    turn_image_map = _build_turn_image_map(image_files, image_spans)

    # ---- 6. Extract attention per turn per layer ----
    result = TrajectoryAttentionResult(
        traj_dir=str(traj_dir),
        image_filenames=image_files,
        layer_indices=list(attn_by_layer.keys()),
    )

    for layer_idx, attn_tensor in attn_by_layer.items():
        # Average across heads: (1, num_heads, L, L) → (L, L)
        attn = attn_tensor[0].float().mean(dim=0).cpu().numpy()

        for resp_span in response_spans:
            turn_result = TurnAttentionResult(
                turn_idx=resp_span.turn_idx,
                layer_idx=layer_idx,
            )

            # Response token rows
            resp_attn = attn[resp_span.token_start:resp_span.token_end, :]
            # Average across response tokens → (L,)
            resp_attn_avg = resp_attn.mean(axis=0)

            # For each image visible at this turn, extract its heatmap
            visible_images = _get_visible_images(
                resp_span.turn_idx, turn_image_map
            )
            for img_filename, img_span in visible_images:
                img_attn = resp_attn_avg[img_span.token_start:img_span.token_end]
                heatmap = attention_to_heatmap(
                    img_attn, img_span.grid_h, img_span.grid_w,
                    target_h, target_w,
                )
                turn_result.image_heatmaps[img_filename] = heatmap

            result.turns.append(turn_result)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rebuild_conversation(
    traj_path: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read messages.json and replace <data_url> placeholders with real image paths.

    Returns:
        (messages, image_filenames) where messages is in Qwen VL chat format
        and image_filenames lists filenames in order of appearance.
    """
    with open(traj_path / "messages.json") as f:
        raw_messages = json.load(f)

    images_dir = traj_path / "images"
    all_image_files = sorted(p.name for p in images_dir.glob("turn_*.png"))

    rebuilt: List[Dict[str, Any]] = []
    image_filenames: List[str] = []
    img_counter = 0

    for msg in raw_messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            rebuilt.append({"role": role, "content": content})
            continue

        new_parts: List[Dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                if img_counter < len(all_image_files):
                    fname = all_image_files[img_counter]
                    img_path = str(images_dir / fname)
                    new_parts.append({
                        "type": "image",
                        "image": f"file://{img_path}",
                    })
                    image_filenames.append(fname)
                    img_counter += 1
            elif part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    new_parts.append({"type": "text", "text": text})
            else:
                new_parts.append(part)

        if role == "assistant":
            text_parts = [p["text"] for p in new_parts if p.get("type") == "text"]
            rebuilt.append({"role": role, "content": " ".join(text_parts)})
        else:
            rebuilt.append({"role": role, "content": new_parts})

    return rebuilt, image_filenames


def _tokenize_conversation(
    messages: List[Dict[str, Any]],
    image_files: List[str],
    manager: ModelManager,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Tokenize the rebuilt conversation using the Qwen2.5-VL processor."""
    from qwen_vl_utils import process_vision_info

    processor = manager.processor

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    device = manager.device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    input_ids_flat = inputs["input_ids"][0]
    seq_len = input_ids_flat.shape[0]
    print(f"  Tokenized: {seq_len} tokens, {len(image_files)} images")

    return inputs, input_ids_flat


def _forward_with_attention(
    inputs: Dict[str, torch.Tensor],
    manager: ModelManager,
) -> Dict[int, torch.Tensor]:
    """
    Run model forward pass and extract attention from the target layers.

    Returns:
        dict mapping layer_idx → attention tensor (1, num_heads, seq_len, seq_len).
    """
    with torch.no_grad():
        outputs = manager.model(**inputs, output_attentions=True)

    result = {}
    for layer_idx in manager.layer_indices:
        attn = outputs.attentions[layer_idx]
        if attn is None:
            print(
                f"  WARNING: Attention for layer {layer_idx} is None, skipping."
            )
            continue
        print(f"  Layer {layer_idx} attention shape: {attn.shape}")
        result[layer_idx] = attn

    return result


def _build_turn_image_map(
    image_filenames: List[str],
    image_spans: List[ImageTokenSpan],
) -> List[Tuple[int, List[Tuple[str, ImageTokenSpan]]]]:
    """
    Build mapping from turn index to list of (filename, span) pairs.

    Turn 0: images from first user message (typically 3).
    Turn N>0: one additional image from the next user message.
    """
    turn_map: List[Tuple[int, List[Tuple[str, ImageTokenSpan]]]] = []

    current_turn = 0
    current_images: List[Tuple[str, ImageTokenSpan]] = []

    for fname, span in zip(image_filenames, image_spans):
        parts = fname.replace(".png", "").split("_")
        file_turn = int(parts[1]) - 1  # 0-indexed

        if file_turn != current_turn and current_images:
            turn_map.append((current_turn, current_images))
            current_images = []
            current_turn = file_turn

        current_images.append((fname, span))

    if current_images:
        turn_map.append((current_turn, current_images))

    return turn_map


def _get_visible_images(
    turn_idx: int,
    turn_image_map: List[Tuple[int, List[Tuple[str, ImageTokenSpan]]]],
) -> List[Tuple[str, ImageTokenSpan]]:
    """Get all images visible at a given turn (cumulative up to that turn)."""
    visible = []
    for map_turn_idx, images in turn_image_map:
        if map_turn_idx <= turn_idx:
            visible.extend(images)
        else:
            break
    return visible
