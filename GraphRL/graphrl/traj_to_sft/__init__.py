"""TrajToSFT phase: convert RL/eval rollouts → LLaMA-Factory dataset.

The only user-extension point in the GraphRL framework. Pick the most
specific base class that fits your env (always single inheritance):

    TrajToSFTModule              ← override run()
    TrajToSFTGraphBase           ← override graph_builder_class + generate_datasets
    TrajToSFTReasoningBase       ← override generate(); reasoning post-step added
    TrajToSFTGraphReasoningBase  ← graph + reasoning post-step

The two reasoning bases run a self-reasoning post-step over the produced
SFT JSONs (controlled by ``traj_to_sft.reasoning.enabled`` in YAML).
See :mod:`graphrl.traj_to_sft.self_reasoning` for the customisation surface.
"""
from graphrl.traj_to_sft.traj_to_sft_base import (
    TrajToSFTModule,
    TrajToSFTPaths,
    load_traj_to_sft_class,
)
from graphrl.traj_to_sft.traj_to_sft_graph_base import TrajToSFTGraphBase
from graphrl.traj_to_sft.traj_to_sft_reasoning_base import TrajToSFTReasoningBase
from graphrl.traj_to_sft.traj_to_sft_graph_reasoning_base import (
    TrajToSFTGraphReasoningBase,
)

__all__ = [
    "TrajToSFTModule",
    "TrajToSFTGraphBase",
    "TrajToSFTReasoningBase",
    "TrajToSFTGraphReasoningBase",
    "TrajToSFTPaths",
    "load_traj_to_sft_class",
]
