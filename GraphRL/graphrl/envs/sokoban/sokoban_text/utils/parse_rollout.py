"""
Parse VAGEN Sokoban rollout JSONL files into (state, action, next_state) transitions.

VAGEN stores each episode as one JSON line with keys:
    "input":  "<|im_start|>system\\n...<|im_end|><|im_start|>user\\n...<|im_end|>..."
    "output": "<|im_start|>assistant\\n...<|im_end|>..."

Concatenating input+output gives the full conversation string, which we parse
into messages and then extract:
  - state grids from user turns  ([Initial Observation] or After that, …)
  - action sequences from assistant turns  (<answer>…</answer>)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── regex patterns ─────────────────────────────────────────────────────────────

_CONV_RE = re.compile(
    r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL
)
_INITIAL_OBS_RE = re.compile(
    r"\[Initial Observation\]:\s*\n(.*?)\nDecide your next action",
    re.DOTALL,
)
_AFTER_OBS_RE = re.compile(
    r"After that, the observation is:\s*\n(.*?)\nDecide your next action",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_grid(text: str) -> str:
    """Strip trailing whitespace from each line for consistent hashing."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _parse_messages(text: str) -> List[Dict[str, str]]:
    text = text.replace("<|endoftext|>", "")
    return [
        {"role": role, "content": content.strip()}
        for role, content in _CONV_RE.findall(text)
    ]


def _extract_grid(content: str, initial: bool) -> Optional[str]:
    pattern = _INITIAL_OBS_RE if initial else _AFTER_OBS_RE
    m = pattern.search(content)
    return _normalize_grid(m.group(1)) if m else None


def _extract_action(content: str) -> Optional[str]:
    m = _ANSWER_RE.search(content)
    return m.group(1).strip() if m else None


# ── public API ────────────────────────────────────────────────────────────────

def parse_transitions(jsonl_path: Path) -> List[Dict[str, str]]:
    """
    Parse one VAGEN rollout JSONL file into a list of transitions.

    Each transition is a dict:
        {"state": str, "action": str, "next_state": str}

    where "action" is the raw content of <answer>…</answer> (may be a
    comma-separated multi-action sequence like "Right, Down").

    Both successful and failed episodes contribute transitions.
    """
    transitions: List[Dict[str, str]] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d  JSON error: %s", jsonl_path, lineno, exc)
                continue

            full_text = data.get("input", "") + data.get("output", "")
            messages = _parse_messages(full_text)

            states: List[str] = []
            actions: List[str] = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    # First user turn → try initial pattern; fall back to "after" pattern
                    is_first = len(states) == 0
                    grid = _extract_grid(content, initial=is_first)
                    if grid is None and is_first:
                        grid = _extract_grid(content, initial=False)
                    if grid:
                        states.append(grid)

                elif role == "assistant":
                    action = _extract_action(content)
                    if action:
                        actions.append(action)

            # Pair up: s0 -e0-> s1 -e1-> s2 …
            for i, action in enumerate(actions):
                if i + 1 < len(states):
                    transitions.append(
                        {
                            "state": states[i],
                            "action": action,
                            "next_state": states[i + 1],
                        }
                    )

    return transitions
