"""Shared data types used by all three pipeline phases (RL / TrajToSFT / SFT).

These are plain dataclasses / enums — *not* an abstract base class. The three
phase classes don't inherit from each other; they just consume the same
``ModuleState`` lifecycle enum and ``ModuleOutput`` chaining record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class ModuleState(Enum):
    IDLE = auto()
    LAUNCHED = auto()
    DONE = auto()
    TERMINATED = auto()
    FAILED = auto()


@dataclass
class ModuleOutput:
    """Output produced by a phase, used to chain phases together."""

    model_path: Optional[str] = None
    data_paths: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
