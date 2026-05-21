"""``TrajToSFTReasoningBase`` — TrajToSFT + reasoning post-step.

Use when your data-generation step is **not** graph-based but you still
want a self-reasoning pass over the produced SFT JSONs (e.g. random-eval
collection, fixed-dataset symlink, filter-only).

Subclass contract — override :meth:`generate` (renamed from
:meth:`TrajToSFTModule.run` so the base class can wrap it with the
reasoning post-step):

    class MyTrajToSFT(TrajToSFTReasoningBase):
        def generate(self) -> None:
            # Read self.paths.rollout_data / self.paths.rl_model
            # Write self.paths.sft_data + dataset_info.json
            ...

After ``generate()`` writes ``dataset_info.json``, the base class checks
``self.config["reasoning"].enabled``. If true, it instantiates a
:class:`graphrl.traj_to_sft.self_reasoning.Reasoner` (default, or
``reasoning.reasoner_cls`` from YAML) and runs it. A ``.reasoning_done``
marker file makes the whole step resume-safe — a crash mid-reasoning
re-runs only the reasoning step on next attempt.

YAML::

    traj_to_sft:
      module: my_pkg.MyTrajToSFT
      reasoning:
        enabled: true
        reasoner_cls: null            # null → default Reasoner
        sglang: { tp_size: 1, dp_size: 8, mem_fraction: 0.80 }
        max_turns: 3
        chat_config: { temperature: 0.2, top_p: 0.9, max_tokens: 2048 }
        # …see Reasoner for the full list of knobs
"""
from __future__ import annotations

from graphrl.traj_to_sft._reasoning_helpers import (
    REASONING_DONE_MARKER,
    maybe_run_reasoning,
    reasoning_done,
    reasoning_enabled,
)
from graphrl.traj_to_sft.traj_to_sft_base import TrajToSFTModule


class TrajToSFTReasoningBase(TrajToSFTModule):
    """TrajToSFT with a self-reasoning post-step.

    Subclasses implement :meth:`generate`; the base class wraps it with
    the reasoning step.
    """

    name = "TrajToSFT(reasoning)"

    # ── lifecycle: same I/O as TrajToSFTModule, plus a reasoning marker ──

    def is_done(self) -> bool:
        if not super().is_done():
            return False
        if not reasoning_enabled(self.config):
            return True
        return reasoning_done(self.paths)

    def run(self) -> None:
        if not super().is_done():
            self.generate()
        maybe_run_reasoning(self.config, self.paths, parent_name=self.name)

    # ── subclass interface ────────────────────────────────────────────────

    def generate(self) -> None:
        """Produce the SFT data — same I/O contract as
        :meth:`TrajToSFTModule.run`: write ``dataset_info.json`` plus
        per-dataset ``.json`` files into ``self.paths.sft_data``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate()."
        )


__all__ = ["REASONING_DONE_MARKER", "TrajToSFTReasoningBase"]
