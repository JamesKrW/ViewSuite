"""Single-turn reasoning annotation: explode multi-turn records into per-turn jobs.

For a record with N assistant turns, we run N annotation jobs. Each job sees
the full conversation prefix up through ``[ASSISTANT i]`` (the target turn,
shown as the last block) but with images filtered to a small "recent" subset
plus the first user turn — the conversation text is preserved fully so the
annotator still has trajectory context, but the visual context is focused on
the decision being annotated. The model is then asked to emit a flat
``<observation>...</observation> ... <action>...</action>`` body (no
``<turn>`` wrapper); :class:`ObsActionChecker`'s flat-mode fallback parses
that natively when ``n_expected=1``.

After all per-turn jobs finish, the per-seed augmentations are grouped by
record and reassembled into the original multi-turn SFT shape — each
assistant message's content is replaced by its augmented body. Per-turn
salvage: if some turns of a record never produced a valid augmentation,
those turns keep their original ``<action>...</action>`` content while the
rest of the record carries the augmented bodies. A record is dropped only
when every one of its turns failed (unless ``keep_unaugmented=true``).

Image policy (``single_turn.recent_k`` in YAML, default 2): keep image
placeholders for user turns ``{1} ∪ {max(1, i-k+1), ..., i}`` and drop
``<image>`` tokens (and their corresponding entries in the flat ``images``
list) from any other user turn so VAGEN's reading-order substitution stays
aligned.

Wired in via ``traj_to_sft.reasoning.reasoner_cls``::

    traj_to_sft:
      reasoning:
        enabled: true
        reasoner_cls: graphrl.traj_to_sft.self_reasoning.single_turn.SingleTurnReasoner
        single_turn:
          recent_k: 2
        # … usual sglang / chat_config / max_turns / suffix knobs …
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .augment import run_vagen_eval_and_collect
from .base import BaseDataset, Datapoint
from .postprocess import _USE_ORIGINAL, _inject_system_suffix
from .reasoner import DEFAULT_PROMPTS_DIR, Reasoner
from .sglang_server import SGLangServer

logger = logging.getLogger(__name__)


# ── exploded-dataset reader ─────────────────────────────────────────────────


class SingleTurnExplodedDataset(BaseDataset):
    """Reads the exploded SFT JSON written by :class:`SingleTurnReasoner`.

    Each record in the exploded JSON has the standard ShareGPT shape
    (``messages`` + ``images``) **plus** a ``_target_action`` field
    carrying the byte-correct expected action for the LAST assistant
    turn shown. We expose that single string as ``assistant_texts`` so
    :class:`ObsActionChecker` only validates one turn — the full prior
    conversation is shown to the annotator as context but is not
    re-checked.
    """

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
            self._records: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, idx: int) -> Datapoint:
        rec = self._records[idx]
        imgs = [self._load(p) for p in rec.get("images", [])]
        # Single expected text = the target action; ignore the messages-derived
        # list so prior assistant history doesn't get re-validated.
        target = rec.get("_target_action")
        if not isinstance(target, str):
            raise ValueError(
                f"SingleTurnExplodedDataset record {idx} missing '_target_action'"
            )
        return Datapoint(
            idx=idx, messages=rec["messages"], images=imgs, assistant_texts=[target],
        )

    def _load(self, rel: str) -> Image.Image:
        img = Image.open(self.image_root / rel).convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        return img


# ── explode + reassemble logic ─────────────────────────────────────────────


@dataclass
class _JobMapping:
    """Seed → (record_idx, turn_idx, n_turns). Persisted next to the
    exploded JSON so resume reads back the exact same mapping the dump
    seeds were generated against."""
    record_idx: int
    turn_idx: int      # 1-based assistant index within the original record
    n_turns: int       # total assistants in the original record


def _kept_user_turns(i: int, recent_k: int) -> set:
    """User turns (1-based) whose images we keep when annotating turn ``i``.

    ``{1} ∪ {max(1, i-k+1), ..., i}``. When i ≤ k the union is just
    ``{1, ..., i}`` — i.e. all user turns up to the current one.
    """
    return {1} | set(range(max(1, i - recent_k + 1), i + 1))


def _build_subrecord(
    messages: List[Dict[str, Any]],
    image_paths: List[str],
    target_assistant_turn: int,
    recent_k: int,
) -> Optional[Dict[str, Any]]:
    """Build one exploded sub-record for the ``target_assistant_turn``-th
    assistant (1-based).

    The returned record's messages contain everything from the original
    up to and including the target assistant block, with the image
    policy applied: kept user turns retain their ``<image>`` placeholders
    and corresponding image paths; dropped user turns have ``<image>``
    stripped from their content and contribute zero images.
    """
    # Index the user/assistant message positions inside the flat messages
    # list. We need the per-user image-slot offsets too.
    user_msg_indices: List[int] = []
    assistant_msg_indices: List[int] = []
    for j, m in enumerate(messages):
        role = m.get("role")
        if role == "user":
            user_msg_indices.append(j)
        elif role == "assistant":
            assistant_msg_indices.append(j)

    if target_assistant_turn < 1 or target_assistant_turn > len(assistant_msg_indices):
        return None
    # Sanity: a well-formed conversation has at least one user turn before
    # the target assistant. If not, skip.
    if not user_msg_indices:
        return None

    # Per user-message <image> placeholder slice into the flat image list.
    img_offsets: List[Tuple[int, int]] = []
    cursor = 0
    for j in user_msg_indices:
        count = messages[j].get("content", "").count("<image>")
        img_offsets.append((cursor, cursor + count))
        cursor += count
    # NOTE: we don't assert cursor == len(image_paths). If the record
    # contains stray images beyond what the user-message placeholders
    # account for, we just ignore the trailing entries — same conservative
    # behaviour as the default ShareGPT path.

    keep_users = _kept_user_turns(target_assistant_turn, recent_k)
    target_assistant_msg_idx = assistant_msg_indices[target_assistant_turn - 1]

    new_messages: List[Dict[str, Any]] = []
    new_images: List[str] = []
    user_turn_seen = 0
    for j, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            new_messages.append(dict(m))
            continue
        if role == "user":
            user_turn_seen += 1
            content = m.get("content", "")
            start, end = img_offsets[user_turn_seen - 1]
            if user_turn_seen in keep_users:
                new_messages.append(dict(m))
                new_images.extend(image_paths[start:end])
            else:
                # Drop placeholders from text *and* the matching image
                # paths from the flat list — VAGEN substitutes by reading
                # order, so both must shrink together.
                stripped = content.replace("<image>", "")
                new_messages.append({"role": "user", "content": stripped})
            continue
        if role == "assistant":
            new_messages.append(dict(m))
            if j == target_assistant_msg_idx:
                # Stop right after the target assistant — anything past it
                # is the future of the trajectory and shouldn't be shown.
                break
            continue
        # Unknown role — pass through so we don't silently lose data.
        new_messages.append(dict(m))

    target_action = messages[target_assistant_msg_idx].get("content", "")
    return {
        "messages": new_messages,
        "images": new_images,
        "_target_action": target_action,
    }


def _explode_records(
    records: List[Dict[str, Any]], recent_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    """Return ``(exploded_records, mapping)``.

    ``mapping[seed]`` is a dict ``{record_idx, turn_idx, n_turns}`` carrying
    the info needed to reassemble the multi-turn output.
    """
    exploded: List[Dict[str, Any]] = []
    mapping: List[Dict[str, int]] = []
    for rec_i, rec in enumerate(records):
        messages = rec.get("messages") or []
        image_paths = list(rec.get("images") or [])
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant == 0:
            continue
        for turn in range(1, n_assistant + 1):
            sub = _build_subrecord(messages, image_paths, turn, recent_k)
            if sub is None:
                continue
            exploded.append(sub)
            mapping.append({"record_idx": rec_i, "turn_idx": turn, "n_turns": n_assistant})
    return exploded, mapping


def _reassemble(
    originals: List[Dict[str, Any]],
    mapping: List[Dict[str, int]],
    per_seed: Dict[int, List[Optional[str]]],
    keep_unaugmented: bool,
    augmented_system_prompt_suffix: Optional[str],
    raw_system_prompt_suffix: Optional[str],
) -> Tuple[List[Dict[str, Any]], Tuple[int, int, int]]:
    """Group per-seed annotations by record and rebuild multi-turn SFT records.

    Per-turn salvage: turns whose seed didn't yield a valid body keep the
    original ``<action>...</action>`` content; the rest of the record uses
    the augmented body. A record is dropped (unless ``keep_unaugmented``)
    only when *every* turn failed.

    Returns ``(records, (n_full, n_partial, n_dropped))``.
    """
    # Group seeds by record_idx.
    by_record: Dict[int, Dict[int, Optional[str]]] = {}
    for seed, info in enumerate(mapping):
        rec_i = info["record_idx"]
        turn = info["turn_idx"]
        bodies = per_seed.get(seed)
        body = bodies[0] if (bodies and len(bodies) > 0) else None
        by_record.setdefault(rec_i, {})[turn] = body

    out: List[Dict[str, Any]] = []
    n_full = n_partial = n_dropped = 0
    for rec_i, rec in enumerate(originals):
        n_assistant = sum(1 for m in rec.get("messages", []) if m.get("role") == "assistant")
        if n_assistant == 0:
            n_dropped += 1
            continue
        slot_by_turn = by_record.get(rec_i, {})
        augmented: List[Optional[str]] = [
            slot_by_turn.get(t) for t in range(1, n_assistant + 1)
        ]
        any_ok = any(b is not None for b in augmented)
        all_ok = all(b is not None for b in augmented)

        if not any_ok:
            if keep_unaugmented:
                out.append(_inject_system_suffix(rec, raw_system_prompt_suffix))
                n_partial += 1
            else:
                n_dropped += 1
            continue

        # At least one turn succeeded — write augmented record with per-turn fallback.
        base = _inject_system_suffix(rec, augmented_system_prompt_suffix)
        new_messages: List[Dict[str, Any]] = []
        a_idx = 0
        for m in base["messages"]:
            if m.get("role") == "assistant":
                aug = augmented[a_idx]
                if aug is _USE_ORIGINAL or aug is None:
                    new_messages.append(dict(m))
                else:
                    new_messages.append({"role": "assistant", "content": aug})
                a_idx += 1
            else:
                new_messages.append(dict(m))
        out.append({**base, "messages": new_messages})
        if all_ok:
            n_full += 1
        else:
            n_partial += 1
    return out, (n_full, n_partial, n_dropped)


# ── reasoner subclass ──────────────────────────────────────────────────────


class SingleTurnReasoner(Reasoner):
    """:class:`Reasoner` variant that explodes records into per-turn jobs.

    Configurable knobs (under ``traj_to_sft.reasoning`` in YAML):

      single_turn.recent_k
        Number of recent user turns whose images are kept (in addition
        to user_1). Default 2 → for turn ``i`` we keep images from
        ``{1, max(1, i-1), i}``. When ``i ≤ recent_k`` all user turns
        up to ``i`` are kept (the union degenerates to ``{1, ..., i}``).

    All other knobs (sglang, chat_config, max_turns, suffixes, prompts,
    n_records_per_target, …) behave identically to the default
    :class:`Reasoner`.
    """

    name = "SingleTurnReasoner"

    # ── overrides ─────────────────────────────────────────────────────────

    def system_prompt_path(self) -> Path:
        p = self.config.get("system_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "single_turn_system.md"

    def user_prompt_path(self) -> Path:
        p = self.config.get("user_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "single_turn_user.md"

    def env_name(self) -> str:
        # Distinct registry name so a process running both the multi-turn
        # and single-turn flows in sequence doesn't collide on the same
        # registered env class.
        return self.config.get("env_name", "GraphRLSingleTurnReasoningEnv")

    def recent_k(self) -> int:
        st = self.config.get("single_turn") or {}
        return int(st.get("recent_k", 2))

    def exploded_path(self, target: str) -> Path:
        """Where the exploded SFT JSON lives. Pinned under ``reasoning_dump/<target>/``
        so it's resume-stable across re-runs."""
        return self.dump_dir(target) / "_exploded.json"

    def mapping_path(self, target: str) -> Path:
        return self.dump_dir(target) / "_mapping.json"

    # ── main entry ────────────────────────────────────────────────────────

    def run(self) -> None:
        targets = self.targets()
        if not targets:
            logger.info("[%s] no reasoning targets in %s; skipping",
                        self.parent_name, self.paths.sft_data)
            return

        self._snapshot_originals(targets)

        model = self.model_path()
        # Same model-path validation as the default Reasoner — single-turn
        # has no reason to differ.
        model_p = Path(model).expanduser()
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
            raise FileNotFoundError(
                f"[{self.parent_name}] reasoning prompts missing: {sys_p}, {usr_p}"
            )

        sglang_cfg = dict(self.config.get("sglang", {}) or {})
        log_root = self.paths.base_dir / "traj_to_sft" / "reasoning_dump"
        log_root.mkdir(parents=True, exist_ok=True)

        recent_k = self.recent_k()
        logger.info(
            "[%s] launching sglang for single-turn reasoning: "
            "model=%s tp=%d dp=%d port=%d recent_k=%d → %d target(s): %s",
            self.parent_name, model,
            int(sglang_cfg.get("tp_size", 1)),
            int(sglang_cfg.get("dp_size", 8)),
            int(sglang_cfg.get("port", 30000)),
            recent_k,
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
                self._augment_target(target, server.base_url, model, sys_p, usr_p)

    # ── per-target step ───────────────────────────────────────────────────

    def _augment_target(
        self,
        target: str,
        base_url: str,
        model: str,
        sys_p: Path,
        usr_p: Path,
    ) -> None:
        snapshot = self.snapshot_path(target)
        if not snapshot.exists():
            logger.warning(
                "[%s] no snapshot for %s at %s; skipping",
                self.parent_name, target, snapshot,
            )
            return

        with open(snapshot, encoding="utf-8") as f:
            originals = json.load(f)

        n_records_cap = self.n_records_for(target)
        if n_records_cap is not None:
            n_records_cap = min(int(n_records_cap), len(originals))
            sliced = originals[:n_records_cap]
        else:
            sliced = originals
        if not sliced:
            logger.warning(
                "[%s] %s: no records to augment (cap=%s, total=%d)",
                self.parent_name, target, n_records_cap, len(originals),
            )
            return

        recent_k = self.recent_k()
        exploded_path = self.exploded_path(target)
        mapping_path = self.mapping_path(target)
        exploded_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume-stable explode: build once and pin to disk; subsequent runs
        # read the same file so seeds in dump_dir/tag_<target>/<seed>/ keep
        # their meaning.
        if exploded_path.exists() and mapping_path.exists():
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
            with open(exploded_path, encoding="utf-8") as f:
                exploded = json.load(f)
            logger.info(
                "[%s] %s: reusing existing exploded view (%d jobs from %d records)",
                self.parent_name, target, len(exploded), len(sliced),
            )
        else:
            exploded, mapping = _explode_records(sliced, recent_k=recent_k)
            if not exploded:
                logger.warning(
                    "[%s] %s: explode produced no jobs; skipping",
                    self.parent_name, target,
                )
                return
            with open(exploded_path, "w", encoding="utf-8") as f:
                json.dump(exploded, f, ensure_ascii=False)
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
            logger.info(
                "[%s] %s: exploded %d records → %d single-turn jobs (recent_k=%d)",
                self.parent_name, target, len(sliced), len(exploded), recent_k,
            )

        # Run vagen against the exploded SFT, using our custom dataset class
        # so assistant_texts is just the target action (one expected turn).
        per_seed = run_vagen_eval_and_collect(
            sft_path=exploded_path,
            image_root=self.image_root(target),
            dump_dir=self.dump_dir(target),
            tag_id=target,
            base_url=base_url,
            model_name=model,
            system_prompt_path=sys_p,
            user_prompt_path=usr_p,
            image_size=self.image_size(),
            max_turns=int(self.config.get("max_turns", 3)),
            max_concurrent_jobs=int(self.config.get("max_concurrent_jobs", 16)),
            max_retries=int(self.config.get("max_retries", 6)),
            chat_config=self.chat_config(),
            salvage_partial=bool(self.config.get("salvage_partial", True)),
            n_records=None,  # the explode + cap was already applied above
            resume=bool(self.config.get("resume", True)),
            env_name=self.env_name(),
            checker_cls=self.config.get("checker_cls"),
            checker_kwargs=self.config.get("checker_kwargs"),
            dataset_cls=(
                "graphrl.traj_to_sft.self_reasoning.single_turn."
                "SingleTurnExplodedDataset"
            ),
            dataset_kwargs=None,
            num_workers=self.num_workers(),
        )

        # Reassemble multi-turn records from per-seed augmentations. Note
        # we reassemble against the ORIGINAL (pre-cap) records so out-of-cap
        # records are dropped (or kept un-augmented if the user opted in).
        out_records, (n_full, n_partial, n_dropped) = _reassemble(
            originals=originals,
            mapping=mapping,
            per_seed=per_seed,
            keep_unaugmented=bool(self.config.get("keep_unaugmented", False)),
            augmented_system_prompt_suffix=self.config.get("augmented_system_prompt_suffix"),
            raw_system_prompt_suffix=self.config.get("raw_system_prompt_suffix"),
        )

        out_path = self.output_path(target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)
        logger.info(
            "[%s] %s: wrote %d records → %s (full=%d partial=%d dropped=%d, jobs=%d)",
            self.parent_name, target, len(out_records), out_path,
            n_full, n_partial, n_dropped, len(mapping),
        )


__all__ = [
    "SingleTurnExplodedDataset",
    "SingleTurnReasoner",
]
