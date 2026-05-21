"""
Map per-image attention to ScanNet mesh vertices, compute overlap statistics.

Core logic:
  1. For each image (target, init, top-down), project mesh vertices to the
     camera image plane and retrieve the attention value at that pixel.
  2. Find the set of vertices visible in at least 2 of the 3 images (any-2 overlap).
  3. Compute mean attention for overlap vs non-overlap vertices.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from view_suite.proxy_analysis.scannet_point_by_turn.visibility import VisibilityComputer


def get_visible_vertices_with_pixels(
    vc: VisibilityComputer,
    c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like vc.get_visible_vertex_indices() but also returns pixel coordinates.

    Returns:
        (vertex_indices, u_pixels, v_pixels) — all 1-D int arrays of same length.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    w2c = np.linalg.inv(c2w)

    # GPU depth rendering
    z_depth = vc._render_z_depth(w2c)

    # Project vertices to camera frame
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    verts_cam = (R @ vc.vertices.T).T + t
    z = verts_cam[:, 2]

    # In front of camera
    mask_front = z > vc.NEAR_CLIP
    idx_front = np.nonzero(mask_front)[0]
    verts_front = verts_cam[mask_front]
    z_front = z[mask_front]

    # Project to pixel coordinates
    proj = (vc.K @ verts_front.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    # Within image bounds
    mask_bounds = (u >= 0) & (u < vc.width) & (v >= 0) & (v < vc.height)
    idx_bounds = idx_front[mask_bounds]
    u_int = u[mask_bounds].astype(np.int32)
    v_int = v[mask_bounds].astype(np.int32)
    z_verts = z_front[mask_bounds]

    np.clip(u_int, 0, vc.width - 1, out=u_int)
    np.clip(v_int, 0, vc.height - 1, out=v_int)

    # Depth test
    rendered_z = z_depth[v_int, u_int]
    mask_visible = np.abs(z_verts - rendered_z) < vc.DEPTH_EPSILON

    return idx_bounds[mask_visible], u_int[mask_visible], v_int[mask_visible]


def compute_vertex_attention(
    attn_1d: np.ndarray,
    grid_h: int,
    grid_w: int,
    vc: VisibilityComputer,
    c2w: np.ndarray,
    img_h: int = 512,
    img_w: int = 512,
) -> Dict[int, float]:
    """
    Map 1-D vision-token attention to mesh vertices.

    For each visible vertex, look up the attention value of the grid cell
    it projects into (no upsampling — direct grid-cell lookup).

    Args:
        attn_1d: (num_tokens,) raw attention scores for one image's tokens.
        grid_h, grid_w: post-merge vision-token grid dimensions.
        vc: VisibilityComputer for the scene.
        c2w: 4x4 camera-to-world matrix for this image.
        img_h, img_w: original rendered image size.

    Returns:
        dict mapping vertex_index → attention value.
    """
    grid = attn_1d.reshape(grid_h, grid_w)

    vis_idx, u_px, v_px = get_visible_vertices_with_pixels(vc, c2w)

    # Map pixel to grid cell
    gc = np.clip((u_px * grid_w / img_w).astype(np.int32), 0, grid_w - 1)
    gr = np.clip((v_px * grid_h / img_h).astype(np.int32), 0, grid_h - 1)

    return {int(idx): float(grid[r, c]) for idx, r, c in zip(vis_idx, gr, gc)}


def compute_overlap_stats(
    vertex_attns: List[Dict[int, float]],
) -> Dict[str, float]:
    """
    Compute mean attention for any-2 overlap vs non-overlap vertices.

    Overlap = vertices visible in at least 2 of the 3 images.
    Non-overlap = vertices visible in exactly 1 image.

    Args:
        vertex_attns: list of 3 dicts (one per image), each mapping
                      vertex_index → attention value.

    Returns:
        dict with keys:
          overlap_mean, non_overlap_mean, n_overlap, n_non_overlap
    """
    # Count how many images each vertex appears in
    from collections import Counter
    counts: Counter = Counter()
    for d in vertex_attns:
        counts.update(d.keys())

    overlap = {v for v, c in counts.items() if c >= 2}
    non_overlap = {v for v, c in counts.items() if c == 1}

    # Overlap: average attention across all images that see this vertex
    overlap_vals = []
    for vidx in overlap:
        for d in vertex_attns:
            if vidx in d:
                overlap_vals.append(d[vidx])

    non_overlap_vals = []
    for vidx in non_overlap:
        for d in vertex_attns:
            if vidx in d:
                non_overlap_vals.append(d[vidx])

    return {
        "overlap_mean": float(np.mean(overlap_vals)) if overlap_vals else 0.0,
        "non_overlap_mean": float(np.mean(non_overlap_vals)) if non_overlap_vals else 0.0,
        "n_overlap": len(overlap),
        "n_non_overlap": len(non_overlap),
    }
