"""
SFT dataset generators for Sokoban text environment.

Four generator functions (all path-based, sampled without replacement):

  path_to_view_direct
      Input:  initial board + concatenated action sequence
      Output: <prediction>final board</prediction>

  path_to_view_mcq
      Input:  initial board + concatenated action sequence + 4 options (A-D)
      Output: letter of correct option (negatives drawn randomly from graph)

  view_to_path
      Input:  initial board + final board  (single-edge transition)
      Output: action sequence

  state_reachable
      Multi-turn conversation:  given s0 and sN, navigate step by step.
      Conversation ends on the last assistant (action) turn.

All generators output ShareGPT message format:
  {"messages": [{"role": "system/user/assistant", "content": "..."}]}
System prompts are randomly sampled from pools of paraphrased variants.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

logger = logging.getLogger(__name__)

_LABELS = "ABCD"

# ── symbol legend (shared across all prompts) ────────────────────────────────

_SYMBOLS = (
    "Symbols: # Wall | _ Floor | O Target | X Box | P Player "
    "| √ Box on Target | S Player on Target"
)

_ACTIONS = "Valid actions: Up, Down, Left, Right (comma-separated for multiple)."

# ── diversified system prompt pools ──────────────────────────────────────────

_FWD_DIRECT_PROMPTS = [
    (
        "You are a Sokoban world model. "
        "Given a board state and an action sequence, predict the resulting board state.\n"
        f"{_SYMBOLS}\n"
        "Respond with ONLY the resulting board wrapped in <prediction>…</prediction>."
    ),
    (
        "You simulate Sokoban board dynamics. "
        "Your task: given the current board and a series of moves, output the final board.\n"
        f"{_SYMBOLS}\n"
        "Wrap your answer in <prediction>…</prediction> tags — nothing else."
    ),
    (
        "Act as a Sokoban state predictor. "
        "Read the board layout and the action sequence, then determine what the board looks like afterward.\n"
        f"{_SYMBOLS}\n"
        "Output only the predicted board inside <prediction>…</prediction>."
    ),
    (
        "You are an expert at mentally simulating Sokoban moves. "
        "Apply the given actions to the board and produce the resulting layout.\n"
        f"{_SYMBOLS}\n"
        "Return the final board state in <prediction>…</prediction> tags only."
    ),
    (
        "Sokoban forward-dynamics task: "
        "given the starting board and a sequence of player actions, predict the outcome.\n"
        f"{_SYMBOLS}\n"
        "Respond with the resulting board enclosed in <prediction>…</prediction>."
    ),
]

_FWD_MCQ_PROMPTS = [
    (
        "You are a Sokoban world model. "
        "Given a board state and an action sequence, identify the correct resulting state.\n"
        f"{_SYMBOLS}\n"
        "Respond with ONLY the letter (A, B, C, or D) of the correct option."
    ),
    (
        "Sokoban multiple-choice challenge: "
        "apply the given actions to the board and pick which option matches the final state.\n"
        f"{_SYMBOLS}\n"
        "Answer with a single letter: A, B, C, or D."
    ),
    (
        "You simulate Sokoban board transitions. "
        "After executing the action sequence, select the option that shows the correct result.\n"
        f"{_SYMBOLS}\n"
        "Reply with only the correct letter (A–D)."
    ),
    (
        "Act as a Sokoban state predictor. "
        "Given the board and moves, choose the matching outcome from four candidates.\n"
        f"{_SYMBOLS}\n"
        "Output just the letter of the correct option."
    ),
    (
        "Predict the Sokoban board after the given actions, then choose the right answer.\n"
        f"{_SYMBOLS}\n"
        "Respond with the correct letter only (A, B, C, or D)."
    ),
]

_INV_PROMPTS = [
    (
        "You are a Sokoban inverse dynamics model. "
        "Given two board states, determine the action sequence that leads from the "
        "first state to the second.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Respond with ONLY the action sequence."
    ),
    (
        "Sokoban action recovery task: "
        "figure out what moves transform the initial board into the resulting board.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Reply with the action sequence only."
    ),
    (
        "Given a before-and-after pair of Sokoban boards, deduce the player actions.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Output just the action sequence, nothing else."
    ),
    (
        "You reverse-engineer Sokoban moves. "
        "Look at the starting board and the ending board, then infer the actions taken.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Respond with the action sequence only."
    ),
    (
        "Act as a Sokoban inverse model. "
        "Determine what sequence of moves was applied to go from the first state to the second.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Answer with just the actions."
    ),
]

_REACH_PROMPTS = [
    (
        "You are a Sokoban navigator. "
        "Given an initial state and a goal state, navigate step by step to reach the goal.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "At each step, output ONLY the action(s) to take."
    ),
    (
        "Navigate a Sokoban board from the start state to the goal state, one step at a time.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Each turn, reply with only the action(s) for that step."
    ),
    (
        "You are a step-by-step Sokoban solver. "
        "You will be shown the current board after each move. Plan your way to the goal.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Respond with only the action(s) at each step."
    ),
    (
        "Sokoban path-following task: "
        "given the initial and target boards, output one action at a time to reach the target.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "At each turn, provide only the action(s)."
    ),
    (
        "Act as a Sokoban agent navigating toward a goal configuration. "
        "You will receive updated board states after each action.\n"
        f"{_SYMBOLS}\n"
        f"{_ACTIONS}\n"
        "Reply with only the action(s) to execute next."
    ),
]


# ── internal helpers ──────────────────────────────────────────────────────────

def _pick_prompt(pool: List[str], rng: random.Random) -> str:
    """Randomly select a system prompt from the pool."""
    return rng.choice(pool)


def _concat_actions(path: List[Dict[str, str]]) -> str:
    """Join all per-step action strings with ', '."""
    return ", ".join(step["action"] for step in path)


def _make_record(system: str, user: str, assistant: str) -> Dict[str, Any]:
    """Build a single-turn ShareGPT message record."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ── generators ────────────────────────────────────────────────────────────────

def generate_path_to_view_direct(
    graph: BaseGraph,
    min_path_len: int = 1,
    max_path_len: int = 3,
    num_samples: int = 2000,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Sample paths, concatenate actions, predict final state.

    Output: ShareGPT messages format.
    """
    if rng is None:
        rng = random.Random()
    paths = graph.sample_paths(min_path_len, max_path_len, num_samples, rng)
    records = []
    for path in paths:
        state = path[0]["from_state"]
        action = _concat_actions(path)
        final_state = path[-1]["to_state"]
        records.append(_make_record(
            system=_pick_prompt(_FWD_DIRECT_PROMPTS, rng),
            user=f"Current state:\n{state}\n\nAction sequence: {action}",
            assistant=f"<prediction>\n{final_state}\n</prediction>",
        ))
    return records


def generate_path_to_view_mcq(
    graph: BaseGraph,
    min_path_len: int = 1,
    max_path_len: int = 3,
    num_samples: int = 2000,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Same as direct but with 4 options (1 correct + 3 negatives from graph).

    Output: ShareGPT messages format.
    """
    if rng is None:
        rng = random.Random()
    if graph.num_nodes < 4:
        logger.warning(
            "[path_to_view_mcq] Graph has fewer than 4 nodes; skipping."
        )
        return []

    paths = graph.sample_paths(min_path_len, max_path_len, num_samples, rng)
    records = []
    for path in paths:
        from_id = path[0]["from_id"]
        to_id = path[-1]["to_id"]

        negatives = graph.get_random_state_texts(3, {from_id, to_id}, rng)
        if len(negatives) < 3:
            continue  # not enough distinct states for 4-option MCQ

        state = path[0]["from_state"]
        action = _concat_actions(path)
        correct = path[-1]["to_state"]

        options = [correct] + negatives
        rng.shuffle(options)
        correct_label = _LABELS[options.index(correct)]

        option_text = "\n\n".join(
            f"{_LABELS[i]})\n{opt}" for i, opt in enumerate(options)
        )
        records.append(_make_record(
            system=_pick_prompt(_FWD_MCQ_PROMPTS, rng),
            user=(
                f"Current state:\n{state}\n\n"
                f"Action sequence: {action}\n\n"
                f"Which of the following is the correct resulting state?\n\n"
                f"{option_text}"
            ),
            assistant=correct_label,
        ))
    return records


def generate_view_to_path(
    graph: BaseGraph,
    num_samples: int = 2000,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Single-edge inverse: given (state, next_state), predict action sequence.

    Samples edges without replacement.

    Output: ShareGPT messages format.
    """
    if rng is None:
        rng = random.Random()
    edges = graph._edges
    if not edges:
        return []
    k = min(num_samples, len(edges))
    sampled = rng.sample(edges, k)
    records = []
    for edge in sampled:
        state = graph.nodes[edge["from"]]["obs_str"]
        next_state = graph.nodes[edge["to"]]["obs_str"]
        action = edge["obs_str"]
        records.append(_make_record(
            system=_pick_prompt(_INV_PROMPTS, rng),
            user=(
                f"Initial state:\n{state}\n\n"
                f"Resulting state:\n{next_state}"
            ),
            assistant=action,
        ))
    return records


def generate_state_reachable(
    graph: BaseGraph,
    min_path_len: int = 3,
    max_path_len: int = 5,
    num_samples: int = 1000,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """
    Multi-turn navigation from s0 to sN along a sampled graph path.

    Conversation ends on the last assistant (action) turn — no trailing
    user confirmation message.

    Output: ShareGPT messages format.
    """
    if rng is None:
        rng = random.Random()
    paths = graph.sample_paths(min_path_len, max_path_len, num_samples, rng)
    if not paths:
        logger.warning("[state_reachable] No paths sampled from graph.")
        return []

    records = []
    for path in paths:
        s0 = path[0]["from_state"]
        sN = path[-1]["to_state"]

        messages = [
            {"role": "system", "content": _pick_prompt(_REACH_PROMPTS, rng)},
            {
                "role": "user",
                "content": (
                    f"Initial state:\n{s0}\n\n"
                    f"Goal state:\n{sN}\n\n"
                    f"Current state:\n{s0}\n\n"
                    "Decide your next action(s)."
                ),
            },
        ]

        for i, step in enumerate(path):
            # assistant outputs the action
            messages.append({"role": "assistant", "content": step["action"]})
            # user shows the resulting state (except after the final step)
            if i < len(path) - 1:
                messages.append({
                    "role": "user",
                    "content": (
                        f"After action '{step['action']}', "
                        f"current state:\n{step['to_state']}\n\n"
                        "Decide your next action(s)."
                    ),
                })
        # conversation ends on the last assistant turn

        records.append({"messages": messages})
    return records
