"""
Load Qwen2.5-VL model and processor for attention extraction.

Uses a memory-efficient trick: only the target attention layer(s) are set to
"eager" mode (which produces explicit attention weights), while all other
layers keep their default SDPA/flash implementation.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


class ModelManager:
    """Manages model and processor lifecycle."""

    def __init__(
        self,
        model_path: str,
        layer_indices: Union[int, List[int]] = -1,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor: Optional[Qwen2_5_VLProcessor] = None

        # Accept int or list
        if isinstance(layer_indices, int):
            self._raw_layer_indices = [layer_indices]
        else:
            self._raw_layer_indices = list(layer_indices)

        self.layer_indices: List[int] = []  # resolved after model load

    def load(self) -> None:
        """Load model and processor, patch target layers for eager attention."""
        print(f"Loading processor from {self.model_path} ...")
        self.processor = Qwen2_5_VLProcessor.from_pretrained(self.model_path)

        print(f"Loading model from {self.model_path} (dtype={self.dtype}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

        # Language model layers: model.model.language_model.layers
        layers = self.model.model.language_model.layers
        num_layers = len(layers)

        # Resolve negative indices
        self.layer_indices = []
        for raw_idx in self._raw_layer_indices:
            idx = raw_idx if raw_idx >= 0 else num_layers + raw_idx
            assert 0 <= idx < num_layers, (
                f"layer_idx {raw_idx} out of range [0, {num_layers})"
            )
            self.layer_indices.append(idx)

        # Patch target layers: give each a config copy with eager attention
        for idx in self.layer_indices:
            target_attn = layers[idx].self_attn
            eager_config = copy.copy(target_attn.config)
            eager_config._attn_implementation = "eager"
            target_attn.config = eager_config

        print(
            f"  Patched layers {self.layer_indices} (of {num_layers}) "
            f"to eager attention"
        )
        print(f"  Model ready on {self.device}")

    @property
    def vision_config(self):
        return self.model.config.vision_config

    @property
    def patch_size(self) -> int:
        return self.vision_config.patch_size  # 14

    @property
    def spatial_merge_size(self) -> int:
        return self.vision_config.spatial_merge_size  # 2

    @property
    def num_heads(self) -> int:
        return self.model.config.num_attention_heads  # 28
