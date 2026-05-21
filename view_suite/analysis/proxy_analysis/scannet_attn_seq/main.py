"""
Full-sequence attention-to-image analysis.

For each trajectory, compute:
  1. Per-turn stats: for each response turn, mean attention fraction to image tokens.
  2. Full-sequence curve: for each token position, attention fraction to image tokens.

The full-sequence curves are averaged across trajectories (aligned by semantic region).

Usage:
    python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.main run \
        --rollout_dir /path/to/rollout \
        --model_path /path/to/model
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import fire

from view_suite.analysis.scannet_attn.model_manager import ModelManager
from view_suite.analysis.scannet_attn.attention_extractor import (
    _rebuild_conversation,
    _tokenize_conversation,
    _forward_with_attention,
)
from view_suite.analysis.scannet_attn.token_image_mapper import (
    find_image_token_spans,
    find_response_token_spans,
    ImageTokenSpan,
    ResponseTokenSpan,
)


def _list_traj_dirs(rollout_dir: str, max_trajs: int = 0) -> List[Path]:
    rd = Path(rollout_dir)
    dirs = sorted([d for d in rd.iterdir() if d.is_dir() and not d.name.startswith(".")])
    dirs = [d for d in dirs if (d / "messages.json").exists()]
    if max_trajs > 0:
        dirs = dirs[:max_trajs]
    return dirs


def _parse_layer_indices(layer_indices) -> List[int]:
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


def _build_image_token_mask(image_spans: List[ImageTokenSpan], seq_len: int) -> np.ndarray:
    """Boolean mask: True for image token positions."""
    mask = np.zeros(seq_len, dtype=bool)
    for span in image_spans:
        mask[span.token_start:span.token_end] = True
    return mask


def _compute_img_attn_fraction_per_position(
    attn_matrix: np.ndarray,
    img_mask: np.ndarray,
) -> np.ndarray:
    """
    For each row (query position), compute fraction of attention going to image tokens.

    Args:
        attn_matrix: (seq_len, seq_len) attention matrix (head-averaged).
        img_mask: (seq_len,) boolean mask for image token positions.

    Returns:
        (seq_len,) array of fractions.
    """
    img_attn = attn_matrix[:, img_mask].sum(axis=1)    # (seq_len,)
    total_attn = attn_matrix.sum(axis=1)                # (seq_len,)
    # Avoid division by zero
    total_attn = np.where(total_attn > 0, total_attn, 1.0)
    return img_attn / total_attn


def _label_sequence_regions(
    seq_len: int,
    image_spans: List[ImageTokenSpan],
    response_spans: List[ResponseTokenSpan],
) -> List[dict]:
    """
    Label each token position with its semantic region.

    Returns a list of region dicts: {type, start, end, index}
    sorted by start position. Types: "other", "image", "response".
    """
    regions = []
    for s in image_spans:
        regions.append({"type": "image", "start": s.token_start, "end": s.token_end,
                        "index": s.image_idx})
    for s in response_spans:
        regions.append({"type": "response", "start": s.token_start, "end": s.token_end,
                        "index": s.turn_idx})
    regions.sort(key=lambda r: r["start"])
    return regions


def _default_output_dir(rollout_dir: str) -> Path:
    """Derive output dir by mirroring rollout path under a sibling ``rollouts_attn_seq`` root.

    Example:
        rollout_dir = ".../rollouts_all/qwen_25_vl_7b/tag_interactive_view_planning"
        →  output   = ".../rollouts_attn_seq/qwen_25_vl_7b/tag_interactive_view_planning"
    """
    rd = Path(rollout_dir).resolve()
    parts = rd.parts
    # Find the rollouts root directory (e.g. "rollouts_all", "rollouts")
    for i, p in enumerate(parts):
        if p.startswith("rollouts"):
            return Path(*parts[:i]) / "rollouts_attn_seq" / Path(*parts[i + 1:])
    # Fallback: place next to rollout_dir
    return rd.parent / "rollouts_attn_seq" / rd.name


class AttnSeqAnalyzer:

    def run(
        self,
        rollout_dir: str,
        model_path: str,
        layer_indices: Union[int, str] = '[0,7,14,21,27]',
        max_trajs: int = 0,
        device: str = "cuda:0",
        output_dir: Optional[str] = None,
    ):
        indices = _parse_layer_indices(layer_indices)
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = _default_output_dir(rollout_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading model (layers {indices}) ...")
        manager = ModelManager(model_path, layer_indices=indices, device=device)
        manager.load()

        traj_dirs = _list_traj_dirs(rollout_dir, max_trajs)
        print(f"Found {len(traj_dirs)} trajectories")

        all_records = []
        t0 = time.time()

        for i, td in enumerate(traj_dirs):
            try:
                messages, image_files = _rebuild_conversation(td)
                inputs, input_ids_flat = _tokenize_conversation(messages, image_files, manager)

                attn_by_layer = _forward_with_attention(inputs, manager)

                image_spans = find_image_token_spans(
                    input_ids_flat, inputs["image_grid_thw"], manager.spatial_merge_size,
                )
                response_spans = find_response_token_spans(
                    input_ids_flat, manager.processor.tokenizer,
                )

                if not response_spans or len(image_spans) < 3:
                    print(f"  [{i+1}] SKIP {td.name}: no response or <3 images")
                    continue

                seq_len = int(input_ids_flat.shape[0])
                img_mask = _build_image_token_mask(image_spans, seq_len)
                n_img_tokens = int(img_mask.sum())
                n_response_turns = len(response_spans)

                record = {
                    "traj": td.name,
                    "seq_len": seq_len,
                    "n_img_tokens": n_img_tokens,
                    "n_images": len(image_spans),
                    "n_response_turns": n_response_turns,
                    "layers": {},
                }

                for layer_idx, attn_tensor in attn_by_layer.items():
                    attn = attn_tensor[0].float().mean(dim=0).cpu().numpy()  # (L, L)

                    # --- Per-position image attention fraction ---
                    per_pos_fraction = _compute_img_attn_fraction_per_position(attn, img_mask)

                    # --- Per response turn stats ---
                    turn_stats = []
                    for resp in response_spans:
                        resp_fractions = per_pos_fraction[resp.token_start:resp.token_end]
                        turn_stats.append({
                            "turn_idx": resp.turn_idx,
                            "mean_img_fraction": float(resp_fractions.mean()),
                            "std_img_fraction": float(resp_fractions.std()),
                            "n_tokens": resp.token_end - resp.token_start,
                        })

                    # --- Region-averaged curve (for cross-trajectory aggregation) ---
                    # Compute mean fraction for: each image block, each response block, and "other"
                    region_curve = []
                    # Images
                    for s in image_spans:
                        frac = per_pos_fraction[s.token_start:s.token_end]
                        region_curve.append({
                            "type": "image", "index": s.image_idx,
                            "mean_fraction": float(frac.mean()),
                        })
                    # Responses
                    for s in response_spans:
                        frac = per_pos_fraction[s.token_start:s.token_end]
                        region_curve.append({
                            "type": "response", "index": s.turn_idx,
                            "mean_fraction": float(frac.mean()),
                        })
                    # Overall non-image, non-response (instruction/special tokens)
                    other_mask = np.ones(seq_len, dtype=bool)
                    for s in image_spans:
                        other_mask[s.token_start:s.token_end] = False
                    for s in response_spans:
                        other_mask[s.token_start:s.token_end] = False
                    if other_mask.any():
                        other_frac = per_pos_fraction[other_mask]
                        region_curve.append({
                            "type": "other", "index": -1,
                            "mean_fraction": float(other_frac.mean()),
                        })

                    record["layers"][str(layer_idx)] = {
                        "turn_stats": turn_stats,
                        "region_curve": region_curve,
                        # Global stats
                        "global_mean_fraction": float(per_pos_fraction.mean()),
                        "response_mean_fraction": float(np.mean([
                            t["mean_img_fraction"] for t in turn_stats
                        ])),
                    }

                all_records.append(record)

                if (i + 1) % 50 == 0 or i == 0:
                    elapsed = time.time() - t0
                    sample_key = str(indices[len(indices) // 2])
                    sample = record["layers"].get(sample_key, {})
                    print(f"  [{i+1}/{len(traj_dirs)}] {elapsed:.0f}s | "
                          f"turns={n_response_turns}, imgs={len(image_spans)}, "
                          f"resp_frac(L{sample_key})="
                          f"{sample.get('response_mean_fraction', 0):.4f}")

            except Exception as e:
                print(f"  [{i+1}] ERROR {td.name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_records:
            print("No records collected!")
            return

        # --- Aggregate ---
        summary = self._aggregate(all_records, indices)

        # Print
        n = len(all_records)
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"{n} trajectories | layers {indices} | {elapsed:.0f}s")
        print(f"{'='*60}")

        for lidx in indices:
            key = str(lidx)
            s = summary[key]
            print(f"\n  Layer {lidx}:")
            print(f"    global_mean_fraction: "
                  f"mean={s['global_mean_fraction']['mean']:.6f} "
                  f"std={s['global_mean_fraction']['std']:.6f}")
            print(f"    Per-turn response→image fraction:")
            for ts in s["per_turn"]:
                print(f"      Turn {ts['turn_idx']}: "
                      f"mean={ts['mean']:.6f} std={ts['std']:.6f} "
                      f"(n={ts['n']})")

        # Save
        payload = {
            "config": {
                "rollout_dir": rollout_dir,
                "model_path": model_path,
                "layer_indices": indices,
            },
            "records": all_records,
            "summary": summary,
        }
        out_path = out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {out_path}")

    def _aggregate(self, all_records, indices):
        summary = {}
        for lidx in indices:
            key = str(lidx)

            global_fracs = []
            # Collect per-turn stats: dict of turn_idx → list of fractions
            turn_fracs: Dict[int, List[float]] = {}

            # Region stats: dict of (type, index) → list of fractions
            region_fracs: Dict[str, List[float]] = {}

            for r in all_records:
                ls = r["layers"].get(key)
                if ls is None:
                    continue
                global_fracs.append(ls["global_mean_fraction"])

                for ts in ls["turn_stats"]:
                    tidx = ts["turn_idx"]
                    turn_fracs.setdefault(tidx, []).append(ts["mean_img_fraction"])

                for rc in ls["region_curve"]:
                    rkey = f"{rc['type']}_{rc['index']}"
                    region_fracs.setdefault(rkey, []).append(rc["mean_fraction"])

            # Per-turn aggregation
            per_turn = []
            for tidx in sorted(turn_fracs.keys()):
                vals = np.array(turn_fracs[tidx])
                per_turn.append({
                    "turn_idx": tidx,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "n": len(vals),
                })

            # Region aggregation
            per_region = {}
            for rkey in sorted(region_fracs.keys()):
                vals = np.array(region_fracs[rkey])
                per_region[rkey] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "n": len(vals),
                }

            gf = np.array(global_fracs)
            summary[key] = {
                "global_mean_fraction": {
                    "mean": float(gf.mean()),
                    "std": float(gf.std()),
                },
                "per_turn": per_turn,
                "per_region": per_region,
                "n_trajectories": len(global_fracs),
            }

        return summary


if __name__ == "__main__":
    fire.Fire(AttnSeqAnalyzer)
