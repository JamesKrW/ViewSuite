# Full-Sequence Image Attention Analysis

Measures how much attention each response token pays to image tokens across an entire multi-turn trajectory, broken down by layer and turn index. Compares RL-trained vs base models to reveal how RL training changes visual grounding patterns.

## Algorithm

### Per-trajectory extraction (`main.py`)

Given a rollout directory and a Qwen2.5-VL model checkpoint:

1. **Conversation replay** — Reconstruct the multi-turn conversation from `messages.json` and image files, then tokenize with the Qwen2.5-VL processor.

2. **Attention extraction** — Run a single forward pass with `output_attentions=True` on specified layers (default: `[0, 7, 14, 21, 27]`). Attention matrices are averaged across heads → `(seq_len, seq_len)`.

3. **Image attention fraction** — For each token position, compute the fraction of its total attention that goes to image tokens: `img_attn_fraction[i] = sum(attn[i, img_positions]) / sum(attn[i, :])`.

4. **Per-turn aggregation** — For each assistant response turn, average the image attention fraction across all response token positions in that turn.

5. **Region curve** — Compute mean attention fraction for each semantic region (image blocks, response blocks, other tokens) for cross-trajectory alignment.

### Cross-trajectory aggregation

Per-turn statistics (mean, std) are aggregated across all trajectories. Trajectories with fewer turns contribute to only the turns they have. A global mean fraction (average over all positions) is also computed per layer.

### Comparison (`compare.py`)

Loads two `results.json` files (RL and Base) and produces:

- **Per-turn line plots** — Response→image attention fraction vs turn index, one subplot per layer, with error bars.
- **Per-region bar plots** — Grouped bars comparing RL vs Base per response turn, one figure per layer.
- **Combined figure** — 2×3 grid: 5 per-turn line subplots + 1 global mean fraction bar chart.

## Architecture

```
main.py         CLI entry (fire.Fire), orchestrates extraction + aggregation
  └─ uses view_suite.analysis.scannet_attn:
       ├─ model_manager.py          Load Qwen2.5-VL, patch layers for eager attention
       ├─ attention_extractor.py    Replay conversation, run forward pass, extract attention
       └─ token_image_mapper.py     Map token indices to image/response spans

compare.py      CLI entry, loads two results.json, generates comparison plots
```

## Output directory

By default, output mirrors the rollout path under a sibling `rollouts_attn_seq/` root:

```
rollouts_all/qwen_25_vl_7b/tag_interactive_view_planning/       (input rollouts)
rollouts_attn_seq/qwen_25_vl_7b/tag_interactive_view_planning/  (output: results.json)
```

This keeps analysis data separate from rollout data. Override with `--output_dir`.

## Usage

```bash
# Step 1: Extract attention for RL (trained) model
python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.main run \
    --rollout_dir /path/to/rollouts_all/qwen_25_vl_7b_trained/tag_interactive_view_planning \
    --model_path /path/to/rl_model

# Step 2: Extract attention for base model
python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.main run \
    --rollout_dir /path/to/rollouts_all/qwen_25_vl_7b/tag_interactive_view_planning \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct

# Step 3: Compare and plot
python -m view_suite.analysis.proxy_analysis.scannet_attn_seq.compare \
    --rl_json /path/to/rollouts_attn_seq/qwen_25_vl_7b_trained/tag_interactive_view_planning/results.json \
    --base_json /path/to/rollouts_attn_seq/qwen_25_vl_7b/tag_interactive_view_planning/results.json \
    --output_dir /path/to/rollouts_attn_seq/compare
```

### Optional arguments for `main.py run`

| Argument | Default | Description |
|---|---|---|
| `--layer_indices` | `'[0,7,14,21,27]'` | Layers to extract attention from (int, list, or JSON string) |
| `--max_trajs` | `0` | Max trajectories to process (0 = all) |
| `--device` | `cuda:0` | GPU device |
| `--output_dir` | auto | Override output directory |

## Output files

### `results.json` (per model)

```json
{
  "config": {"rollout_dir": "...", "model_path": "...", "layer_indices": [0,7,14,21,27]},
  "records": [
    {
      "traj": "20260316-045757-...",
      "seq_len": 8192,
      "n_img_tokens": 3456,
      "n_images": 11,
      "n_response_turns": 10,
      "layers": {
        "0": {
          "turn_stats": [{"turn_idx": 0, "mean_img_fraction": 0.09, "std_img_fraction": 0.02, "n_tokens": 128}],
          "region_curve": [{"type": "response", "index": 0, "mean_fraction": 0.09}],
          "global_mean_fraction": 0.35,
          "response_mean_fraction": 0.07
        }
      }
    }
  ],
  "summary": {
    "0": {
      "global_mean_fraction": {"mean": 0.35, "std": 0.05},
      "per_turn": [{"turn_idx": 0, "mean": 0.09, "std": 0.03, "n": 530}],
      "per_region": {"response_0": {"mean": 0.09, "std": 0.03, "n": 530}},
      "n_trajectories": 530
    }
  }
}
```

### Comparison plots

| File | Description |
|---|---|
| `combined.png` | 2×3 grid: per-turn lines (5 layers) + global fraction bar chart |
| `per_turn_lines.png` | Expanded per-turn line plots, one subplot per layer |
| `per_region_layer{N}.png` | Grouped bar chart per layer, RL vs Base per response turn |
