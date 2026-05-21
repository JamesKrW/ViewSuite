"""Turn a vagen rollout dump directory into a reasoning-augmented SFT JSON.

Per-record outcome (after vagen finishes):

  * **Fully successful rollout** — every turn's checker passed at some
    attempt. Use the ``augmented`` list from the success info verbatim.
  * **Partial success (max_turns exhausted with some turns still failing)**
    — when ``salvage_partial`` is on, fall back per-turn: for turns that
    passed the checker at least once we use their augmented body; for
    turns that never passed we put the **original** assistant content
    back unchanged. The student model trains on a record that mixes
    reasoning-wrapped turns and plain-action turns rather than dropping
    the whole record.
  * **No usable rollout at all** — drop the record (or keep the
    un-augmented original if ``keep_unaugmented``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel: this turn's augmented body is missing — fall back to the
# original assistant content at apply time.
_USE_ORIGINAL = None


def collect_augmented(
    tag_dump_dir: Path,
    *,
    salvage_partial: bool = True,
) -> Dict[int, List[Optional[str]]]:
    """Scan a vagen tag-dump dir and return ``seed → list[Optional[augmented_body]]``.

    Each entry in the returned per-seed list corresponds to one assistant
    turn. ``None`` means "use the original assistant content for this turn"
    (only emitted when ``salvage_partial=True``).
    """
    out: Dict[int, List[Optional[str]]] = {}
    if not tag_dump_dir.is_dir():
        return out
    for rollout_entry in tag_dump_dir.iterdir():
        if not rollout_entry.is_dir():
            continue
        metrics_path = rollout_entry / "metrics.json"
        if not metrics_path.is_file():
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        seed = metrics.get("seed")
        if seed is None:
            continue
        infos = metrics.get("infos", [])
        # Prefer a fully-successful rollout when available.
        full = _find_full_augmented(infos)
        if full is not None:
            out[int(seed)] = list(full)
            continue
        # Otherwise salvage per-turn: use the best augmented body we ever
        # saw for each turn across all attempts; ``None`` for turns that
        # never produced a valid body.
        if salvage_partial:
            partial = _salvage_partial(infos)
            if partial is not None:
                out[int(seed)] = partial
    return out


def _find_full_augmented(infos: List[Dict[str, Any]]) -> Optional[List[str]]:
    for info in reversed(infos):
        if (
            isinstance(info, dict)
            and info.get("success")
            and isinstance(info.get("augmented"), list)
        ):
            return [str(x) for x in info["augmented"]]
    return None


def _salvage_partial(infos: List[Dict[str, Any]]) -> Optional[List[Optional[str]]]:
    """Look across all attempts, take the best-ever augmented body per turn.

    Returns ``[body_or_None, body_or_None, ...]`` — one entry per assistant
    turn, in 1-based order. Returns ``None`` if we can't even infer the
    expected number of turns (no per_turn anywhere).
    """
    n_turns: int = 0
    best: Dict[int, str] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        per_turn = info.get("per_turn")
        if not isinstance(per_turn, list):
            continue
        n_turns = max(n_turns, len(per_turn))
        for entry in per_turn:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("idx")
            if not isinstance(idx, int):
                continue
            if entry.get("ok") and isinstance(entry.get("augmented"), str):
                # First good attempt for this turn wins (we iterate forward, so
                # ``setdefault`` keeps the earliest = the one the env actually
                # accepted on its way to ``success``).
                best.setdefault(idx, entry["augmented"])
    if n_turns == 0:
        return None
    # 1-based ``idx`` → 0-based list. Fill missing with ``None``.
    return [best.get(i + 1, _USE_ORIGINAL) for i in range(n_turns)]


def build_sft(
    sft_path: Path,
    augmented: Dict[int, List[Optional[str]]],
    out_path: Path,
    keep_unaugmented: bool = False,
    augmented_system_prompt_suffix: Optional[str] = None,
    raw_system_prompt_suffix: Optional[str] = None,
) -> int:
    """Write a new SFT JSON whose assistant turns are reasoning-augmented
    where possible and fall back to the original content otherwise.

    Optional per-class prompt suffixes are appended to the SYSTEM message
    of each record so the SFT model learns *conditional* reasoning:
    records that ran reasoning get one instruction, records that stayed raw
    get another. Use this to teach "always reason" vs "reasoning optional"
    behaviour from a single mixed dataset. If a record has no system
    message, one is prepended with the suffix as its content.

    Returns the number of records written.
    """
    with open(sft_path, encoding="utf-8") as f:
        records = json.load(f)

    new_records: List[Dict[str, Any]] = []
    n_full = n_partial = n_dropped = 0
    for idx, rec in enumerate(records):
        aug = augmented.get(idx)
        if aug is None:
            if keep_unaugmented:
                new_records.append(_inject_system_suffix(rec, raw_system_prompt_suffix))
                n_partial += 1
            else:
                n_dropped += 1
            continue
        new_rec = _apply(rec, aug, augmented_system_prompt_suffix)
        if new_rec is None:
            # Length mismatch (can happen if rollout used different turn count).
            if keep_unaugmented:
                new_records.append(_inject_system_suffix(rec, raw_system_prompt_suffix))
                n_partial += 1
            else:
                n_dropped += 1
            continue
        if any(a is _USE_ORIGINAL for a in aug):
            n_partial += 1
        else:
            n_full += 1
        new_records.append(new_rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %d records to %s (full=%d, partial-salvage=%d, dropped=%d, total_in=%d)",
        len(new_records), out_path, n_full, n_partial, n_dropped, len(records),
    )
    return len(new_records)


def _apply(
    record: Dict[str, Any],
    augmented: List[Optional[str]],
    system_prompt_suffix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Replace assistant contents with augmented bodies, falling back to
    the original ``content`` for turns whose augmented body is ``None``.
    Optionally append ``system_prompt_suffix`` to the system message
    (or prepend a new system message containing it if none exists).

    Returns ``None`` if the augmented list length doesn't match the
    record's assistant-turn count.
    """
    base = _inject_system_suffix(record, system_prompt_suffix)
    new_messages: List[Dict[str, Any]] = []
    a_idx = 0
    for m in base["messages"]:
        if m["role"] == "assistant":
            if a_idx >= len(augmented):
                return None
            aug = augmented[a_idx]
            if aug is _USE_ORIGINAL:
                # Salvage: keep the original assistant content as-is — it
                # already contains its own ``<action>...</action>`` tag,
                # so the structure is intact, just no reasoning prefix.
                new_messages.append(dict(m))
            else:
                new_messages.append({"role": "assistant", "content": aug})
            a_idx += 1
        else:
            new_messages.append(dict(m))
    if a_idx != len(augmented):
        return None
    return {**base, "messages": new_messages}


def _inject_system_suffix(
    record: Dict[str, Any], suffix: Optional[str],
) -> Dict[str, Any]:
    """Append ``suffix`` to the system message (or prepend a new system
    message containing the suffix if the record has none). No-op if
    ``suffix`` is falsy. Returns a new record dict; input is not mutated.
    """
    if not suffix:
        return record
    new_messages: List[Dict[str, Any]] = []
    suffix_applied = False
    for m in record["messages"]:
        if m["role"] == "system" and not suffix_applied:
            new_messages.append({
                "role": "system",
                "content": m["content"] + "\n\n" + suffix,
            })
            suffix_applied = True
        else:
            new_messages.append(dict(m))
    if not suffix_applied:
        # No system message in this record — prepend one with just the suffix.
        new_messages.insert(0, {"role": "system", "content": suffix.strip()})
    return {**record, "messages": new_messages}
