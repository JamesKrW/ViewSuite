"""``TrajToSFTGraphReasoningBase`` — graph-based TrajToSFT + reasoning post-step.

Use when your data-generation step IS graph-based (build graph from VAGEN
rollouts, sample paths, emit SFT records) AND you want a self-reasoning
pass over the produced SFT JSONs.

Subclass contract — same as :class:`TrajToSFTGraphBase` (override
:meth:`graph_builder_class` and :meth:`generate_datasets`); the reasoning
post-step is added by this base.

    from graphrl.traj_to_sft import TrajToSFTGraphReasoningBase

    class MyTrajToSFT(TrajToSFTGraphReasoningBase):
        def graph_builder_class(self):
            return MyGraphBuilder
        def generate_datasets(self, graph, images_dir):
            return {"my_dataset": (records, fmt_override)}

Same YAML knobs as :class:`TrajToSFTReasoningBase` — see its docstring.
"""
from __future__ import annotations

from graphrl.traj_to_sft._reasoning_helpers import (
    REASONING_DONE_MARKER,
    maybe_run_reasoning,
    reasoning_done,
    reasoning_enabled,
)
from graphrl.traj_to_sft.traj_to_sft_graph_base import TrajToSFTGraphBase


class TrajToSFTGraphReasoningBase(TrajToSFTGraphBase):
    """:class:`TrajToSFTGraphBase` + self-reasoning post-step.

    Subclass interface is identical to :class:`TrajToSFTGraphBase` —
    override :meth:`graph_builder_class` and :meth:`generate_datasets`.
    """

    name = "TrajToSFT(graph+reasoning)"

    def is_done(self) -> bool:
        if not super().is_done():
            return False
        if not reasoning_enabled(self.config):
            return True
        return reasoning_done(self.paths)

    def run(self) -> None:
        # Graph-base.run() builds graph + runs generate_datasets + writes
        # dataset_info.json. Skip if already done (resume-safe).
        if not super().is_done():
            super().run()
        maybe_run_reasoning(self.config, self.paths, parent_name=self.name)


__all__ = ["REASONING_DONE_MARKER", "TrajToSFTGraphReasoningBase"]
