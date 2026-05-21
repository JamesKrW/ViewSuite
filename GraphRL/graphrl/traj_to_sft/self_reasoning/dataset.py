"""ShareGPT SFT dataset reader used by the reasoning env.

The SFT file is a list of dicts with ``{"messages": [...], "images": [...]}``.
Image paths are resolved against ``image_root`` (defaults to the SFT file's dir).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .base import BaseDataset, Datapoint


class ShareGPTDataset(BaseDataset):
    def __init__(
        self,
        sft_path: str,
        image_root: Optional[str] = None,
        image_size: Optional[List[int]] = None,
    ):
        self.sft_path = Path(sft_path)
        self.image_root = Path(image_root) if image_root else self.sft_path.parent
        self.image_size = tuple(image_size) if image_size else None
        with open(self.sft_path, encoding="utf-8") as f:
            self._records: List[Dict] = json.load(f)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, idx: int) -> Datapoint:
        rec = self._records[idx]
        imgs = [self._load(p) for p in rec.get("images", [])]
        assist = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
        return Datapoint(idx=idx, messages=rec["messages"], images=imgs, assistant_texts=assist)

    def _load(self, rel: str) -> Image.Image:
        img = Image.open(self.image_root / rel).convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        return img
