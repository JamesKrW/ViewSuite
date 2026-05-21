"""
Overlay attention heatmaps on original images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a scalar heatmap onto an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) float32 attention scores (will be normalized to [0, 1]).
        alpha: blending factor for the heatmap overlay.
        colormap: matplotlib colormap name.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    # Normalize heatmap to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min < 1e-12:
        norm = np.zeros_like(heatmap)
    else:
        norm = (heatmap - h_min) / (h_max - h_min)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(norm)[:, :, :3]  # (H, W, 3) float in [0, 1]
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Alpha blend
    img_f = image.astype(np.float32)
    ovl_f = colored_uint8.astype(np.float32)
    blended = (1 - alpha) * img_f + alpha * ovl_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_heatmap_image(
    image_path: str,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
    colormap: str = "jet",
    add_colorbar: bool = True,
) -> None:
    """
    Load an image, overlay heatmap, and save with optional colorbar.

    Args:
        image_path: path to original image.
        heatmap: (H, W) float32 attention heatmap (same size as image).
        output_path: where to save.
        alpha: overlay transparency.
        colormap: matplotlib colormap.
        add_colorbar: whether to add a colorbar annotation.
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    # Resize heatmap if size mismatch
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap_pil = Image.fromarray(
            ((heatmap - heatmap.min()) / max(heatmap.max() - heatmap.min(), 1e-12) * 255)
            .astype(np.uint8),
            mode="L",
        )
        heatmap_pil = heatmap_pil.resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR
        )
        heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0

    if not add_colorbar:
        blended = overlay_heatmap(img, heatmap, alpha, colormap)
        Image.fromarray(blended).save(output_path)
        return

    # With colorbar: use matplotlib
    blended = overlay_heatmap(img, heatmap, alpha, colormap)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.imshow(blended)
    ax.set_axis_off()

    # Add a thin colorbar on the right
    h_min, h_max = heatmap.min(), heatmap.max()
    norm = matplotlib.colors.Normalize(vmin=h_min, vmax=h_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Attention", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
