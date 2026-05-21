"""Self-reasoning post-step for the TrajToSFT phase.

Pipeline overview::

  iter_XXX/
  └── sft_data/
      ├── dataset_info.json     ← from the data-generation step
      ├── multi_turn_action_gen.json  ← raw assistant turns (`<action>...</action>`)
      └── …                     ← any other generators

After data generation finishes, a :class:`Reasoner` (default class lives
here, customisable via ``reasoning.reasoner_cls`` in YAML) launches a
local sglang server hosting the iter's just-trained ``rl_model`` and
walks every produced JSON, replacing each assistant ``<action>...</action>``
turn with a ``<turn><observation>...</observation> ... <action>...</action></turn>``
block. The next SFT phase trains on the augmented data, so reasoning
quality tracks RL progress.

Use the framework bases :class:`graphrl.traj_to_sft.TrajToSFTReasoningBase`
or :class:`graphrl.traj_to_sft.TrajToSFTGraphReasoningBase` to inherit
this behaviour with a single line of `class X(...)`.
"""

from .augment import augment_sft_json, run_vagen_eval_and_collect
from .base import BaseChecker, BaseDataset, Datapoint, TurnCheck, import_by_path
from .checker import ObsActionChecker
from .dataset import ShareGPTDataset
from .env import ReasoningEnv
from .reasoner import DEFAULT_PROMPTS_DIR, Reasoner, resolve_reasoner_cls
from .sglang_server import SGLangServer
from .mcq_reasoning import (
    MCQChecker,
    MCQDatapoint,
    MCQExplodedDataset,
    MCQReasoner,
    MCQReasoningEnv,
)
from .single_turn import SingleTurnExplodedDataset, SingleTurnReasoner

__all__ = [
    "augment_sft_json",
    "BaseChecker",
    "BaseDataset",
    "Datapoint",
    "DEFAULT_PROMPTS_DIR",
    "import_by_path",
    "ObsActionChecker",
    "Reasoner",
    "ReasoningEnv",
    "resolve_reasoner_cls",
    "run_vagen_eval_and_collect",
    "SGLangServer",
    "MCQChecker",
    "MCQDatapoint",
    "MCQExplodedDataset",
    "MCQReasoner",
    "MCQReasoningEnv",
    "ShareGPTDataset",
    "SingleTurnExplodedDataset",
    "SingleTurnReasoner",
    "TurnCheck",
]
