"""Rule-based checker for the ``<observation>...<action>`` reasoning format.

Each augmented assistant turn is wrapped in ``<turn index="N">...</turn>``
and must contain, in this order:

    <observation>short visual description</observation>
    free-form thought (no tags)
    <action>EXACT original action content</action>

The checker extracts per-turn bodies and validates action equivalence.

**Tier-1 salvage (``salvage_action_mismatch=True``).** When the model's
turn block is well-formed in every other respect (``<observation>`` present,
``<action>`` present, length under the cap) but the ``<action>`` content
doesn't byte-match the original assistant's action, we silently substitute
the byte-correct original action into the model's body and mark the turn
``ok=True, salvaged=True``. Most observed action mismatches are stray junk
appended to a correct primitive (e.g. ``move_right|answer(...)``) — the
surrounding observation/thought are still on-topic, so swapping just the
action recovers fully-augmented data without retrying. For cases where
the model genuinely intended a different action, the worst case is one
slightly-misaligned record per occurrence.

For genuinely broken bodies (no ``<observation>``, no ``<action>``, body
too long, missing ``<turn>`` block) the checker still rejects — the
postprocess Tier-2 salvage falls back to the original assistant content
for those turns after ``max_turns`` are exhausted.
"""
from __future__ import annotations

import re
from typing import List

from .base import BaseChecker, TurnCheck

_TURN_RE = re.compile(r'<turn\s+index="(\d+)"\s*>(.*?)</turn>', re.DOTALL)
_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
_ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def _parse_flat(reply: str) -> List[str]:
    """Parse a reply that lacks ``<turn>`` wrappers but still contains
    contiguous ``<observation>...</observation> {thought} <action>...</action>``
    chunks.

    Returns one body string per chunk, in document order. A chunk is
    ``[start of <observation> tag → end of </action> tag]``. Orphan
    observations (no following action before the next ``<observation>``)
    and orphan actions (no preceding observation since the last chunk)
    are skipped silently.

    Used as a fallback when the model has been SFT-trained on body-only
    augmented turns (no ``<turn>`` wrapper) and consequently produces the
    flat shape during the next iter's reasoning annotation step.
    """
    bodies: List[str] = []
    pos = 0
    while True:
        obs = _OBS_RE.search(reply, pos)
        if obs is None:
            break
        act = _ACT_RE.search(reply, obs.end())
        if act is None:
            break
        next_obs = _OBS_RE.search(reply, obs.end())
        if next_obs is not None and next_obs.start() < act.start():
            # Another <observation> showed up before any <action> — this
            # observation is orphaned, advance to the next candidate.
            pos = next_obs.start()
            continue
        bodies.append(reply[obs.start():act.end()])
        pos = act.end()
    return bodies


class ObsActionChecker(BaseChecker):
    """Validates ``<turn>`` blocks containing ``<observation>`` + thought + ``<action>``."""

    def __init__(
        self,
        max_turn_chars: int = 1500,
        salvage_action_mismatch: bool = True,
        accept_flat: bool = True,
    ):
        self.max_turn_chars = int(max_turn_chars)
        self.salvage_action_mismatch = bool(salvage_action_mismatch)
        self.accept_flat = bool(accept_flat)

    def check(self, reply: str, expected_assistant_texts: List[str]) -> List[TurnCheck]:
        n_expected = len(expected_assistant_texts)
        strict = {int(m.group(1)): m.group(2) for m in _TURN_RE.finditer(reply)}

        # Strict parsing is only usable when it cleanly covers every expected
        # turn AND no captured body contains a stray ``<turn`` open — the
        # latter would mean the non-greedy regex absorbed several
        # malformed-nested turns into one giant body (which we observed in
        # iter_001 reasoning runs after the SFT model lost the wrapper).
        strict_good = (
            set(range(1, n_expected + 1)).issubset(strict.keys())
            and all("<turn" not in strict[i] for i in range(1, n_expected + 1))
        )

        if strict_good:
            blocks = strict
        elif self.accept_flat:
            # Flat-format fallback: scan for ``<observation>...<action>``
            # chunks anywhere in the reply (ignoring any ``<turn>`` tag
            # noise) and pair them up positionally with the expected turns.
            flat = _parse_flat(reply)
            blocks = {i + 1: body for i, body in enumerate(flat)}
        else:
            blocks = strict
        out: List[TurnCheck] = []
        for i, original in enumerate(expected_assistant_texts, start=1):
            body = blocks.get(i)
            if body is None:
                out.append(TurnCheck(i, False, f'missing <turn index="{i}">', None))
                continue
            if len(body) > self.max_turn_chars:
                out.append(TurnCheck(
                    i, False,
                    f"turn too long ({len(body)} > {self.max_turn_chars})",
                    None,
                ))
                continue
            obs = _OBS_RE.search(body)
            act = _ACT_RE.search(body)
            if obs is None:
                out.append(TurnCheck(i, False, "missing <observation>", None))
                continue
            if act is None:
                out.append(TurnCheck(i, False, "missing <action>", None))
                continue
            expected_action = self._extract_action(original)
            got_action = act.group(1).strip()
            if got_action != expected_action:
                if self.salvage_action_mismatch:
                    # Tier-1 salvage: substitute the byte-correct action into
                    # the model's body. Use a lambda for the replacement so
                    # backslashes / special regex chars in ``expected_action``
                    # are taken verbatim. ``count=1`` only swaps the first
                    # ``<action>...</action>`` occurrence (the canonical slot).
                    salvaged_body = _ACT_RE.sub(
                        lambda _m: f"<action>{expected_action}</action>",
                        body, count=1,
                    )
                    out.append(TurnCheck(
                        i, True, "", salvaged_body.strip(), salvaged=True,
                    ))
                    continue
                out.append(TurnCheck(
                    i, False,
                    f"action mismatch: got {got_action!r}, expected {expected_action!r}",
                    None,
                ))
                continue
            out.append(TurnCheck(i, True, "", body.strip()))
        return out

    @staticmethod
    def _extract_action(original_assistant_text: str) -> str:
        m = _ACT_RE.search(original_assistant_text)
        return m.group(1).strip() if m else original_assistant_text.strip()
