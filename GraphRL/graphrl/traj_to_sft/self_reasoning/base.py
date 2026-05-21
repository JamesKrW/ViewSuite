"""Abstract interfaces for the reasoning-augmentation pipeline.

Split responsibilities:
  * ``BaseDataset``  — read an SFT file and access a datapoint by index.
  * ``BaseChecker``  — validate a model reply against the expected actions
    and extract the per-turn augmented texts.

Concrete implementations live alongside (``dataset.py`` / ``checker.py``).
Swap them via fully-qualified class names in the YAML config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class Datapoint:
    """A single SFT record flattened for the env."""
    idx: int
    messages: List[Dict[str, str]]       # ShareGPT messages (role/content)
    images: List[Image.Image]            # loaded PIL images in flat order
    assistant_texts: List[str]           # original assistant message contents


class BaseDataset(ABC):
    """Reads an SFT file and provides random-access Datapoints."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, idx: int) -> Datapoint: ...


@dataclass
class TurnCheck:
    idx: int                    # 1-based assistant turn index
    ok: bool
    error: str
    augmented: Optional[str]    # full augmented assistant text when ok
    salvaged: bool = False      # ``ok=True`` was reached via salvage (e.g.
                                # action-content swap), not a clean pass


class BaseChecker(ABC):
    """Validates a model reply and returns per-turn extraction."""

    @abstractmethod
    def check(self, reply: str, expected_assistant_texts: List[str]) -> List[TurnCheck]: ...

    @staticmethod
    def all_ok(results: List[TurnCheck]) -> bool:
        return bool(results) and all(r.ok for r in results)

    @staticmethod
    def feedback(results: List[TurnCheck]) -> str:
        lines = ["Some turns are invalid. Regenerate the FULL output following the format exactly.", ""]
        for r in results:
            if not r.ok:
                lines.append(f"- Turn {r.idx}: {r.error}")
        return "\n".join(lines)


def import_by_path(path: str):
    """Resolve 'pkg.mod.ClassName' to the class object."""
    mod_path, _, name = path.rpartition(".")
    if not mod_path:
        raise ValueError(f"Expected 'module.Class', got {path!r}")
    import importlib
    return getattr(importlib.import_module(mod_path), name)
