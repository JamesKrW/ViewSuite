# Point Cloud Coverage Analysis by Turn

Measures how much of a ScanNet scene's mesh an agent observes over its trajectory, computed as the cumulative union of visible mesh vertices across turns.

## Algorithm

### Per-pose visibility (`visibility.py`)

Given a camera pose and a triangle mesh, determine which mesh vertices are visible:

1. **GPU depth rendering** — Render a 512x512 depth map from the pose using Open3D `OffscreenRenderer`. The renderer returns ray distance (Euclidean), which is converted to camera-frame z-depth via a precomputed `ray_dir_z` correction map.

2. **Vertex projection** — Transform all N mesh vertices from world coordinates to camera coordinates: `verts_cam = R @ verts + t` (using w2c matrix). Then project to pixel coordinates via scaled intrinsics: `u = fx * x/z + cx`, `v = fy * y/z + cy`.

3. **Frustum culling** — Discard vertices behind the camera (`z <= 0.01m`) or outside the image bounds.

4. **Depth test** — For each remaining vertex, compare its projected z-depth against the rendered depth at its pixel location. If `|vertex_z - rendered_z| < 5cm`, the vertex is **visible** (on or near the surface). Otherwise it is **occluded** (behind another surface).

5. **Output** — A set of integer vertex indices (into the mesh's vertex array) that are visible from this pose.

### Per-trajectory accumulation (`scene_processor.py`)

For each trajectory:

- **Turn 0**: `visible = init_view_visible ∪ top_down_view_visible`
- **Turn k** (k=1,2,...): `visible = visible ∪ turn_k_visible`

Since the mesh vertex array is fixed, vertex indices are stable identifiers — `set` union directly gives the cumulative unique visible count without any point-cloud registration.

### Turn pose extraction (`trajectory_parser.py`)

Camera poses are extracted from `messages.json` via regex matching on `"Current camera 6-DoF"` blocks:

- Each user message after an agent action contains a `Current camera` SE(3) pose (tx, ty, tz, rx, ry, rz in degrees, Euler XYZ).
- If an action has a format error, the camera stays at the same pose — this naturally produces zero increment.
- Once the assistant outputs `answer(...)`, all subsequent messages are discarded.
- `init_view` and `top_down_view` c2w matrices come from the evaluation JSONL (exact 4x4 matrices).

### Intrinsics

The environment uses `fix_intrinsics=True`, so all trajectories share the same default intrinsics (`fx=462.073, fy=617.312, cx=255.326, cy=259.135` for 512x512). The `MeshRenderer._scale_K_with_letterbox()` method is applied to match the renderer's letterbox scaling.

## Architecture

```
main.py                 CLI entry (fire.Fire), orchestrates pipeline
  ├─ trajectory_parser.py   Parse rollout dirs → TrajectoryInfo (poses)
  ├─ scene_processor.py     Worker: load mesh once, process all trajs per scene
  │    └─ visibility.py     GPU depth render + vertex projection + depth test
  ├─ aggregator.py          Per-turn mean/std/min/max statistics
  └─ plotter.py             matplotlib plots (cumulative, increment, coverage, all-trajs)
```

**Multiprocessing**: Trajectories are grouped by `scene_id`. Each worker process loads one mesh and creates one GPU `OffscreenRenderer`, then processes all trajectories for that scene sequentially. This minimizes mesh loading overhead (29 scenes for 530 trajectories, or 286 scenes for 3400+ trajectories).

**Checkpointing**: Results are saved incrementally to `checkpoint.jsonl` (one line per trajectory) as each scene completes. On re-run, already-completed scenes are skipped. The checkpoint is deleted after final outputs are written.

**JSONL auto-detection**: If no `--jsonl_path` is provided, the parser reads `env_config.jsonl_path` from each trajectory's `metrics.json` and resolves the path automatically (with fallback path remapping).

## Usage

```bash
# Run ALL models — output to rollouts_pointcloud_coverage/
python -m view_suite.analysis.proxy_analysis.pointcloud_coverage.main run_all \
    --rollouts_dir /path/to/rollouts \
    --scannet_dir /path/to/scannet \
    --n_workers 8

# Single model — output to <output_dir> or <rollout_dir>/coverage_analysis/
python -m view_suite.analysis.proxy_analysis.pointcloud_coverage.main run \
    --rollout_dir /path/to/model/tag_interactive_view_planning \
    --scannet_dir /path/to/scannet \
    --n_workers 8

# Re-plot from saved results
python -m view_suite.analysis.proxy_analysis.pointcloud_coverage.main plot_only \
    --result_json /path/to/results.json

# Compare multiple models on one figure
python -m view_suite.analysis.proxy_analysis.pointcloud_coverage.compare \
    /path/to/model_a_output \
    /path/to/model_b_output \
    --output_dir ./compare_output \
    --labels "Model A,Model B"

# Split by success/fail and generate comparison plots
python -m view_suite.analysis.proxy_analysis.pointcloud_coverage.main split_by_success \
    --rollout_dir /path/to/rollouts
```

## Outputs

| File | Description |
|------|-------------|
| `results.json` | Per-trajectory cumulative counts, increments, coverage ratios |
| `summary.json` | Per-turn aggregated statistics (mean, std, min, max) |
| `results.csv` | Flat CSV with one row per trajectory |
| `avg_cumulative.png` | Mean cumulative visible vertices curve (±1 std) |
| `avg_increment.png` | Mean per-turn new vertices bar chart (±1 std) |
| `avg_coverage_ratio.png` | Mean coverage ratio curve (visible / total vertices) |
| `all_trajectories.png` | All individual cumulative curves overlaid |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEPTH_EPSILON` | 0.05m | Depth test tolerance for vertex visibility |
| `NEAR_CLIP` | 0.01m | Minimum z-depth (discard vertices behind camera) |
| `width x height` | 512x512 | Depth rendering resolution |
| `n_workers` | 8 | Number of parallel worker processes |
