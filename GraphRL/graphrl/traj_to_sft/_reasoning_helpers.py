"""Private helpers shared between :class:`TrajToSFTReasoningBase` and
:class:`TrajToSFTGraphReasoningBase`. Not part of the public API.
"""
from __future__ import annotations

from typing import Any, Dict

from graphrl.traj_to_sft.traj_to_sft_base import TrajToSFTPaths

REASONING_DONE_MARKER = ".reasoning_done"


def reasoning_enabled(config: Dict[str, Any]) -> bool:
    cfg = config.get("reasoning") or {}
    return bool(cfg.get("enabled", False))


def reasoning_done(paths: TrajToSFTPaths) -> bool:
    return (paths.sft_data / REASONING_DONE_MARKER).exists()


def maybe_run_reasoning(
    config: Dict[str, Any],
    paths: TrajToSFTPaths,
    parent_name: str,
) -> None:
    """Run the reasoning post-step if enabled and not already done.

    Idempotent — drops a ``.reasoning_done`` marker on success so re-runs are
    safe (the next attempt would skip immediately).
    """
    if not reasoning_enabled(config):
        return
    if reasoning_done(paths):
        return

    from graphrl.traj_to_sft.self_reasoning import resolve_reasoner_cls

    cfg = config.get("reasoning") or {}
    cls = resolve_reasoner_cls(cfg.get("reasoner_cls"))
    cls(cfg, paths, parent_name=parent_name).run()
    (paths.sft_data / REASONING_DONE_MARKER).write_text("done\n", encoding="utf-8")
