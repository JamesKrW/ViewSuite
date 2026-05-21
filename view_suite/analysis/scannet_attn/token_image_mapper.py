"""
Map vision token indices in the input sequence back to image pixel regions.

Qwen2.5-VL pipeline:
  Original image (H, W)
    → resize to nearest multiple of (patch_size * spatial_merge_size)
    → split into (H/14) x (W/14) patches  [pre-merge grid, given by image_grid_thw]
    → spatial merge 2x2 → (H/28) x (W/28) vision tokens in the LLM sequence
    → each token covers 28x28 pixels in the *resized* image

Token layout in input_ids:
  ... <|vision_start|> [324 image_pad tokens] <|vision_end|> ...
  Tokens are in row-major order of the post-merge grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


# Special token IDs (Qwen2.5-VL)
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655


@dataclass
class ImageTokenSpan:
    """Describes one image's vision tokens in the input sequence."""

    image_idx: int          # 0-based image index (order of appearance)
    token_start: int        # inclusive start position in input_ids
    token_end: int          # exclusive end position in input_ids
    grid_h: int             # post-merge grid height
    grid_w: int             # post-merge grid width
    num_tokens: int         # = grid_h * grid_w

    @property
    def token_indices(self) -> List[int]:
        return list(range(self.token_start, self.token_end))


@dataclass
class ResponseTokenSpan:
    """Describes one assistant response's tokens in the input sequence."""

    turn_idx: int           # 0-based turn index
    token_start: int        # inclusive
    token_end: int          # exclusive


def find_image_token_spans(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> List[ImageTokenSpan]:
    """
    Locate each image's vision tokens in the input sequence.

    Args:
        input_ids: (seq_len,) token id tensor.
        image_grid_thw: (num_images, 3) tensor from processor — each row is
                        (T, H_pre_merge, W_pre_merge).
        spatial_merge_size: 2 for Qwen2.5-VL.

    Returns:
        List of ImageTokenSpan, one per image, in order of appearance.
    """
    ids = input_ids.tolist()
    spans: List[ImageTokenSpan] = []
    search_from = 0

    for img_i in range(image_grid_thw.shape[0]):
        t, h_pre, w_pre = image_grid_thw[img_i].tolist()
        h_post = int(h_pre) // spatial_merge_size
        w_post = int(w_pre) // spatial_merge_size
        num_tokens = int(t) * h_post * w_post

        # Find vision_start token
        try:
            vs_pos = ids.index(VISION_START_ID, search_from)
        except ValueError:
            raise RuntimeError(
                f"Cannot find vision_start for image {img_i} "
                f"(search_from={search_from})"
            )

        token_start = vs_pos + 1  # first vision token is right after vision_start
        token_end = token_start + num_tokens

        spans.append(ImageTokenSpan(
            image_idx=img_i,
            token_start=token_start,
            token_end=token_end,
            grid_h=h_post,
            grid_w=w_post,
            num_tokens=num_tokens,
        ))
        search_from = token_end

    return spans


def find_response_token_spans(
    input_ids: torch.Tensor,
    tokenizer,
) -> List[ResponseTokenSpan]:
    """
    Locate assistant response tokens in the input sequence.

    Qwen2.5-VL chat template uses:
      <|im_start|>assistant\n ... <|im_end|>

    We find each such block and return the token range of the response content.
    """
    ids = input_ids.tolist()

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token = tokenizer.encode("assistant", add_special_tokens=False)

    spans: List[ResponseTokenSpan] = []
    turn_idx = 0
    i = 0

    while i < len(ids):
        if ids[i] == im_start_id:
            # Check if next tokens spell "assistant"
            after = ids[i + 1: i + 1 + len(assistant_token)]
            if after == assistant_token:
                # Content starts after "assistant\n" — find the newline
                content_start = i + 1 + len(assistant_token)
                # Skip newline token(s) right after "assistant"
                while content_start < len(ids) and ids[content_start] in (
                    tokenizer.convert_tokens_to_ids("\n"),
                ):
                    content_start += 1

                # Find im_end
                try:
                    content_end = ids.index(im_end_id, content_start)
                except ValueError:
                    content_end = len(ids)

                if content_end > content_start:
                    spans.append(ResponseTokenSpan(
                        turn_idx=turn_idx,
                        token_start=content_start,
                        token_end=content_end,
                    ))
                    turn_idx += 1
                i = content_end + 1
                continue
        i += 1

    return spans


def attention_to_heatmap(
    attn_scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    target_h: int = 512,
    target_w: int = 512,
) -> np.ndarray:
    """
    Reshape 1D attention scores over vision tokens into a 2D spatial heatmap
    and upsample to the original image resolution.

    Args:
        attn_scores: (num_tokens,) attention scores for one image's vision tokens.
        grid_h, grid_w: post-merge grid dimensions.
        target_h, target_w: output heatmap resolution (original image size).

    Returns:
        (target_h, target_w) float32 heatmap.
    """
    grid = attn_scores.reshape(grid_h, grid_w)

    # Bilinear upsample using PIL (avoids torch dependency here)
    from PIL import Image

    # Normalize to 0-255 for PIL, then resize, then back to float
    grid_min, grid_max = grid.min(), grid.max()
    if grid_max - grid_min < 1e-12:
        return np.zeros((target_h, target_w), dtype=np.float32)

    grid_norm = (grid - grid_min) / (grid_max - grid_min)
    img = Image.fromarray((grid_norm * 255).astype(np.uint8), mode="L")
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    heatmap = np.array(img_resized, dtype=np.float32) / 255.0

    # Restore original scale
    heatmap = heatmap * (grid_max - grid_min) + grid_min
    return heatmap
