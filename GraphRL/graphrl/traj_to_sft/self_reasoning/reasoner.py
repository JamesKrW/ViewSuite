"""Reasoner: customisable post-step that augments SFT data with self-reasoning.

This is the **only user-extension point** for the reasoning post-step.
A subclass can override any of the small hooks below to:

  * choose which produced datasets get augmented (:meth:`targets`),
  * supply a different validity rule (:meth:`make_checker`),
  * supply a different SFT-record reader (:meth:`make_dataset_kwargs`),
  * point at different prompts (:meth:`system_prompt_path`,
    :meth:`user_prompt_path`),
  * use a different model than the iter's just-trained ``rl_model``
    (:meth:`model_path`).

The framework's :class:`TrajToSFTReasoningBase` and
:class:`TrajToSFTGraphReasoningBase` invoke a ``Reasoner`` after their
data-generation step. The default class works whenever the produced SFT
data follows the ShareGPT shape and assistant turns are wrapped in
``<action>...</action>`` (so :class:`ObsActionChecker` can validate the
augmented output). For other formats — write a subclass.

Subclass selection is config-driven::

    traj_to_sft:
      reasoning:
        enabled: true
        reasoner_cls: my_pkg.MyReasoner   # null → use this Reasoner
        # …all other reasoning knobs (sglang, prompts, chat_config, …)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from graphrl.traj_to_sft.traj_to_sft_base import TrajToSFTPaths

from .augment import augment_sft_json
from .base import BaseChecker, BaseDataset
from .checker import ObsActionChecker
from .dataset import ShareGPTDataset
from .sglang_server import SGLangServer

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Reasoner:
    """Default reasoner. Subclass + override hooks to customise behaviour.

    All hooks read from ``self.config`` (= the ``traj_to_sft.reasoning``
    sub-block) by default, so most customisation can happen in YAML; subclass
    only when you need different *logic*.
    """

    name = "Reasoner"

    # ── construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        config: Dict[str, Any],
        paths: TrajToSFTPaths,
        parent_name: str = "TrajToSFT",
    ):
        self.config = config
        self.paths = paths
        self.parent_name = parent_name

    # ── hooks (override to customise) ─────────────────────────────────────

    def targets(self) -> List[str]:
        """Names of datasets in ``sft_data/`` to augment.

        Default: every ``*.json`` in ``sft_data/`` except ``dataset_info.json``.
        Override or set ``reasoning.targets`` in YAML to limit.
        """
        explicit = self.config.get("targets")
        if explicit:
            return [t for t in explicit if (self.paths.sft_data / f"{t}.json").exists()]
        return sorted(
            p.stem for p in self.paths.sft_data.glob("*.json")
            if p.name != "dataset_info.json"
        )

    def make_checker(self) -> BaseChecker:
        """Build the per-record validity checker. Default: :class:`ObsActionChecker`.

        Override (or set ``reasoning.checker_cls`` to a dotted path) for envs
        whose assistant content isn't ``<action>...</action>``.
        """
        from .base import import_by_path
        cls_spec: Optional[str] = self.config.get("checker_cls")
        kwargs: Dict[str, Any] = self.config.get("checker_kwargs") or {}
        if cls_spec:
            cls = import_by_path(cls_spec)
        else:
            cls = ObsActionChecker
        return cls(**kwargs)

    def make_dataset_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs forwarded to the dataset constructor.

        Default returns an empty dict; the augment loop appends ``sft_path``,
        ``image_root``, ``image_size`` per call. Override or set
        ``reasoning.dataset_cls`` / ``dataset_kwargs`` to use a custom reader.
        """
        return self.config.get("dataset_kwargs") or {}

    def dataset_cls_spec(self) -> Optional[str]:
        """Dotted path of a custom :class:`BaseDataset` subclass, or None."""
        return self.config.get("dataset_cls")

    def system_prompt_path(self) -> Path:
        """Path to the system prompt fed to the model."""
        p = self.config.get("system_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "system.md"

    def user_prompt_path(self) -> Path:
        """Path to the user (rules) prompt fed to the model."""
        p = self.config.get("user_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "user.md"

    def model_path(self) -> str:
        """Path/name of the model the sglang server hosts.

        Default: ``self.paths.rl_model`` — the iter's just-trained RL ckpt,
        so reasoning quality tracks RL progress. Override to use a fixed
        teacher model (e.g. a stronger external model).
        """
        explicit = self.config.get("model_path")
        return str(explicit) if explicit else str(self.paths.rl_model)

    def chat_config(self) -> Optional[Dict[str, Any]]:
        return self.config.get("chat_config")

    def num_workers(self) -> int:
        """Number of subprocess workers VAGEN spawns for the rollout loop.

        Default 1 → single-process asyncio (legacy behaviour). Set higher
        when the per-job CPU work (image preprocessing, JSON, regex) makes
        the orchestrator GIL-bound and the GPU starves; each worker runs
        an independent asyncio loop against the shared sglang server.
        Total in-flight = ``num_workers * max_concurrent_jobs``.
        """
        return int(self.config.get("num_workers", 1))

    def image_size(self) -> Optional[List[int]]:
        return self.config.get("image_size")

    def env_name(self) -> str:
        """VAGEN registry name for our :class:`ReasoningEnv`. Must be unique
        per-process; the default works for one Reasoner running at a time."""
        return self.config.get("env_name", "GraphRLReasoningEnv")

    def image_root(self, target: str) -> Path:
        """Where the dataset's relative ``images: [...]`` paths resolve from.

        Default: ``sft_data/`` itself (active-explore generators write
        ``sft_data/images/...`` and reference them as ``images/abc.png``).
        Override if your dataset uses a different layout.
        """
        return self.paths.sft_data

    def output_path(self, target: str) -> Path:
        """Where to write the augmented JSON. Default: ``sft_data/<target>.json``,
        i.e. overwrite what the data-gen step produced — that's the file
        LLaMA-Factory will pick up via ``dataset_info.json``.

        The original (un-augmented) version is preserved at
        :meth:`snapshot_path` before reasoning runs.
        """
        return self.paths.sft_data / f"{target}.json"

    def snapshot_dir(self) -> Path:
        """Where to preserve un-augmented copies of the SFT JSONs.

        Default: ``iter_XXX/traj_to_sft/sft_data_old/`` — TrajToSFT-owned
        scratch space, separate from the LF-consumed ``sft_data/``.
        """
        return self.paths.base_dir / "traj_to_sft" / "sft_data_old"

    def snapshot_path(self, target: str) -> Path:
        """Per-target snapshot path used as the augment input."""
        return self.snapshot_dir() / f"{target}.json"

    def dump_dir(self, target: str) -> Path:
        """Where vagen ``run_eval`` dumps per-record rollouts (resume-safe)."""
        return self.paths.base_dir / "traj_to_sft" / "reasoning_dump" / target

    def n_records_for(self, target: str) -> Optional[int]:
        """How many records of ``target`` to actually augment.

        Resolves in order:
          1. ``reasoning.n_records_per_target.<target>`` (per-target override)
          2. ``reasoning.n_records`` (single value applied to every target)
          3. ``None`` (= run all records in the snapshot)

        Records beyond the cap simply aren't rolled out; with default
        ``keep_unaugmented=false`` they're dropped from the final SFT data.
        Use this when the graph build is expected to produce more records
        than you want to spend reasoning compute on.
        """
        per = self.config.get("n_records_per_target") or {}
        if target in per and per[target] is not None:
            return int(per[target])
        cap = self.config.get("n_records")
        return int(cap) if cap is not None else None

    # ── snapshot helpers (override only if you want a different policy) ──

    def _snapshot_originals(self, targets: List[str]) -> None:
        """Copy ``sft_data/<target>.json`` → ``sft_data_old/<target>.json``
        for each target whose snapshot doesn't already exist.

        Idempotent. After this method, every target has a pristine copy of
        the un-augmented data at :meth:`snapshot_path`, regardless of how
        many partial reasoning runs preceded.
        """
        import shutil
        old_dir = self.snapshot_dir()
        old_dir.mkdir(parents=True, exist_ok=True)
        for target in targets:
            src = self.paths.sft_data / f"{target}.json"
            dst = self.snapshot_path(target)
            if not src.exists():
                logger.warning("[%s] no source for %s at %s; skipping snapshot",
                               self.parent_name, target, src)
                continue
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            logger.info("[%s] snapshot %s → %s", self.parent_name, src, dst)

    # ── main entry (don't usually override) ───────────────────────────────

    def run(self) -> None:
        targets = self.targets()
        if not targets:
            logger.info("[%s] no reasoning targets in %s; skipping",
                        self.parent_name, self.paths.sft_data)
            return

        # Preserve un-augmented originals BEFORE we overwrite them. Idempotent —
        # on resume the existing snapshots stay intact, so augmenting always
        # reads from a pristine copy regardless of partial-overwrite state.
        self._snapshot_originals(targets)

        model = self.model_path()
        # Local path → must be a usable HF model dir. HF Hub id (e.g.
        # ``Qwen/Qwen2.5-VL-7B-Instruct``, no leading ``/`` or ``./`` and
        # not present on disk) is fine — sglang downloads it on launch.
        model_p = Path(model).expanduser()
        # Check the original string for ``./`` / ``../`` prefix — Path()
        # normalizes those away, so test before constructing the Path.
        is_local = (
            model_p.is_absolute()
            or model_p.exists()
            or str(model).startswith(("./", "../"))
        )
        if is_local:
            if not model_p.is_dir() or not (model_p / "config.json").exists():
                raise RuntimeError(
                    f"[{self.parent_name}] model_path {model} is not a usable HF model dir. "
                    "Did RL get skipped this iter without pre-placing rl_model?"
                )
        else:
            logger.info(
                "[%s] model_path %r looks like an HF Hub id; sglang will fetch it.",
                self.parent_name, model,
            )

        sys_p = self.system_prompt_path()
        usr_p = self.user_prompt_path()
        if not sys_p.exists() or not usr_p.exists():
            raise FileNotFoundError(f"[{self.parent_name}] reasoning prompts missing: {sys_p}, {usr_p}")

        sglang_cfg = dict(self.config.get("sglang", {}) or {})
        log_root = self.paths.base_dir / "traj_to_sft" / "reasoning_dump"
        log_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%s] launching sglang for reasoning: model=%s tp=%d dp=%d port=%d → %d target(s): %s",
            self.parent_name, model,
            int(sglang_cfg.get("tp_size", 1)),
            int(sglang_cfg.get("dp_size", 8)),
            int(sglang_cfg.get("port", 30000)),
            len(targets), targets,
        )

        with SGLangServer(
            model_path=model,
            port=int(sglang_cfg.get("port", 30000)),
            tp_size=int(sglang_cfg.get("tp_size", 1)),
            dp_size=int(sglang_cfg.get("dp_size", 8)),
            mem_fraction=float(sglang_cfg.get("mem_fraction", 0.80)),
            ready_timeout=int(sglang_cfg.get("ready_timeout", 1800)),
            extra_args=list(sglang_cfg.get("extra_args", []) or []),
            log_path=str(log_root / "sglang_server.log"),
        ) as server:
            for target in targets:
                # Always read from the snapshot (pristine original) and write
                # the augmented version into sft_data/. That makes resume
                # deterministic even if the previous run partially overwrote
                # ``sft_data/<target>.json``.
                snapshot = self.snapshot_path(target)
                logger.info("[%s] augmenting %s → %s",
                            self.parent_name, snapshot, self.output_path(target))
                augment_sft_json(
                    sft_path=snapshot,
                    image_root=self.image_root(target),
                    output_path=self.output_path(target),
                    dump_dir=self.dump_dir(target),
                    tag_id=target,
                    base_url=server.base_url,
                    model_name=model,
                    system_prompt_path=sys_p,
                    user_prompt_path=usr_p,
                    image_size=self.image_size(),
                    max_turns=int(self.config.get("max_turns", 3)),
                    max_concurrent_jobs=int(self.config.get("max_concurrent_jobs", 16)),
                    max_retries=int(self.config.get("max_retries", 6)),
                    chat_config=self.chat_config(),
                    keep_unaugmented=bool(self.config.get("keep_unaugmented", False)),
                    salvage_partial=bool(self.config.get("salvage_partial", True)),
                    n_records=self.n_records_for(target),
                    resume=bool(self.config.get("resume", True)),
                    env_name=self.env_name(),
                    checker_cls=self.config.get("checker_cls"),
                    checker_kwargs=self.config.get("checker_kwargs"),
                    augmented_system_prompt_suffix=self.config.get("augmented_system_prompt_suffix"),
                    raw_system_prompt_suffix=self.config.get("raw_system_prompt_suffix"),
                    num_workers=self.num_workers(),
                )


# ── resolver helper used by the reasoning bases ─────────────────────────────


def resolve_reasoner_cls(spec: Optional[str]) -> Type[Reasoner]:
    """Resolve a dotted-path reasoner class spec, or fall back to :class:`Reasoner`."""
    if not spec:
        return Reasoner
    from .base import import_by_path
    cls = import_by_path(spec)
    if not isinstance(cls, type) or not issubclass(cls, Reasoner):
        raise TypeError(f"reasoner_cls {spec!r} must be a subclass of Reasoner")
    return cls
