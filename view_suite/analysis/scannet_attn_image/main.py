"""
Compute attention from query_pos to image tokens, for each layer.

query_pos = last token before a chosen response (first or last).
image tokens = all images that appear before query_pos.

Usage:
    # First response (default)
    python -m view_suite.analysis.scannet_attn_image.main run \
        --rollout_dir /path/to/rollout \
        --model_path /path/to/model

    # Last response
    python -m view_suite.analysis.scannet_attn_image.main run \
        --rollout_dir /path/to/rollout \
        --model_path /path/to/model \
        --response last
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import fire

from view_suite.proxy_analysis.scannet_attn.model_manager import ModelManager
from view_suite.proxy_analysis.scannet_attn.attention_extractor import (
    _rebuild_conversation,
    _tokenize_conversation,
    _forward_with_attention,
)
from view_suite.proxy_analysis.scannet_attn.token_image_mapper import (
    find_image_token_spans,
    find_response_token_spans,
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


class AttnImageAnalyzer:

    def run(
        self,
        rollout_dir: str,
        model_path: str,
        layer_indices: Union[int, str] = '[0,7,14,21,27]',
        response: str = "first",
        max_trajs: int = 0,
        device: str = "cuda:0",
        output_dir: Optional[str] = None,
    ):
        """
        Compute attention from query_pos to image tokens for all layers.

        Args:
            response: "first" or "last" — which assistant response to use as
                      query position.  Default "first".
        """
        assert response in ("first", "last"), f"response must be 'first' or 'last', got {response!r}"
        indices = _parse_layer_indices(layer_indices)
        default_suffix = "attn_image_analysis" if response == "first" else "attn_image_analysis_last"
        out_dir = Path(output_dir) if output_dir else Path(rollout_dir) / default_suffix
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Response mode: {response}")

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

                chosen_resp = response_spans[0] if response == "first" else response_spans[-1]
                query_pos = chosen_resp.token_start - 1

                # Use all image spans whose tokens are before query_pos
                relevant_spans = [s for s in image_spans if s.token_end <= query_pos]
                if not relevant_spans:
                    print(f"  [{i+1}] SKIP {td.name}: no image spans before query_pos")
                    continue

                total_img_tokens = sum(s.num_tokens for s in relevant_spans)
                seq_len = int(input_ids_flat.shape[0])

                record = {
                    "traj": td.name,
                    "seq_len": seq_len,
                    "total_img_tokens": total_img_tokens,
                    "n_images": len(relevant_spans),
                    "query_pos": int(query_pos),
                    "response_mode": response,
                    "layers": {},
                }

                for layer_idx, attn_tensor in attn_by_layer.items():
                    attn = attn_tensor[0].float().mean(dim=0).cpu().numpy()
                    query_attn = attn[query_pos, :]

                    img_attn_vals = []
                    for span in relevant_spans:
                        img_attn_vals.append(query_attn[span.token_start:span.token_end])
                    all_img_attn = np.concatenate(img_attn_vals)

                    attn_sum = float(all_img_attn.sum())
                    total_attn = float(query_attn.sum())

                    record["layers"][str(layer_idx)] = {
                        "attn_sum": attn_sum,
                        "attn_mean": float(all_img_attn.mean()),
                        "attn_max": float(all_img_attn.max()),
                        "attn_min": float(all_img_attn.min()),
                        "total_attn": total_attn,
                        "img_attn_fraction": attn_sum / total_attn if total_attn > 0 else 0,
                    }

                all_records.append(record)

                if (i + 1) % 50 == 0 or i == 0:
                    elapsed = time.time() - t0
                    sample_key = str(indices[len(indices) // 2])
                    sample = record["layers"].get(sample_key, {})
                    print(f"  [{i+1}/{len(traj_dirs)}] {elapsed:.0f}s | "
                          f"img_tokens={total_img_tokens}, seq_len={seq_len}, "
                          f"fraction(L{sample_key})={sample.get('img_attn_fraction', 0):.4f}")

            except Exception as e:
                print(f"  [{i+1}] ERROR {td.name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_records:
            print("No records collected!")
            return

        # Aggregate per layer
        summary = {}
        for lidx in indices:
            key = str(lidx)
            vals = {
                "attn_sum": [], "attn_mean": [], "attn_max": [], "attn_min": [],
                "total_attn": [], "img_attn_fraction": [],
            }
            img_tokens_set = set()
            for r in all_records:
                ls = r["layers"].get(key)
                if ls is None:
                    continue
                for k in vals:
                    vals[k].append(ls[k])
                img_tokens_set.add(r["total_img_tokens"])

            layer_summary = {}
            for k, v in vals.items():
                arr = np.array(v)
                layer_summary[k] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            layer_summary["n_trajectories"] = len(vals["attn_sum"])
            layer_summary["img_token_counts"] = sorted(img_tokens_set)
            summary[key] = layer_summary

        # Print summary
        n = len(all_records)
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"{n} trajectories | layers {indices} | {elapsed:.0f}s")
        print(f"{'='*60}")
        for lidx in indices:
            key = str(lidx)
            s = summary[key]
            print(f"\n  Layer {lidx}:")
            print(f"    img_token_counts: {s['img_token_counts']}")
            for metric in ["attn_mean", "attn_max", "attn_min", "attn_sum",
                           "img_attn_fraction", "total_attn"]:
                m = s[metric]
                print(f"    {metric}: mean={m['mean']:.6e} std={m['std']:.6e} "
                      f"[{m['min']:.6e}, {m['max']:.6e}]")

        # Save
        payload = {
            "config": {
                "rollout_dir": rollout_dir,
                "model_path": model_path,
                "layer_indices": indices,
                "response": response,
            },
            "records": all_records,
            "summary": summary,
        }
        out_path = out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    fire.Fire(AttnImageAnalyzer)
