"""Subprocess entry point for ``vagen.evaluate.run_eval``.

The ``GraphRL/VAGEN`` checkout used by the venv lacks two pieces that the
random-action eval YAML relies on:

  1. The ``ScannetTool`` env class — declared in GraphRL's own
     ``graphrl/configs/vagen_configs/env_registry.yaml`` (loaded by Hydra
     during the RL phase, but not by ``python -m vagen.evaluate.run_eval``).
  2. The ``random_response`` / ``random_navigation`` backend adapters and
     client factory — present in the upstream ``viewsuite/VAGEN`` but not in
     ``GraphRL/VAGEN``. We carry copies of the adapters in this package and
     wire up the matching client factory here.

This shim registers all of the above into VAGEN's registries, then defers
entirely to ``vagen.evaluate.run_eval.main``. argv passes through unchanged.
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from vagen.envs.registry import register_env
from vagen.evaluate.registry import register_client

# Side-effect imports: these decorate themselves into vagen.evaluate.registry's
# adapter table, registering the "random_response" and "random_navigation"
# backends.
from graphrl.envs.viewsuite.viewsuite_random_sft_rl import (  # noqa: F401
    random_navigation_adapter,
    random_response_adapter,
)

# This file lives at graphrl/envs/viewsuite/viewsuite_random_sft_rl/, so
# parents[3] is the ``graphrl/`` package root.
_GRAPHRL_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "configs" / "vagen_configs" / "env_registry.yaml"
)


@register_client("random_response", "random_navigation")
def _build_random_client(cfg: Dict[str, Any]) -> None:
    """No real client object needed for the random adapters."""
    return None


def _register_graphrl_envs() -> None:
    if not _GRAPHRL_REGISTRY.is_file():
        return
    cfg = yaml.safe_load(_GRAPHRL_REGISTRY.read_text()) or {}
    for name, dotted in (cfg.get("env_registry") or {}).items():
        module_name, _, class_name = dotted.rpartition(".")
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
            register_env(name, getattr(module, class_name))
        except Exception as exc:  # noqa: BLE001 - best-effort registration
            print(
                f"[run_eval_subprocess] skip {name} ({dotted}): {exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    _register_graphrl_envs()
    runpy.run_module("vagen.evaluate.run_eval", run_name="__main__")
