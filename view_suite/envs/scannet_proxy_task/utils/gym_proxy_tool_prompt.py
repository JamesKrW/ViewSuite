# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
from typing import List

from view_suite.envs.utils.parse_utils import get_format_instruction

def build_example_prompt(example_count: int) -> str:
    """
    Return formatted few-shot examples for the active-exploration pose estimation task.
    The output uses the free_think format and includes [image] placeholders after each status block.
    """
    examples = [
        # ---------------- Example 1 ----------------
        """Example 1:
USER: Now please estimate the camera pose of the target view  .
You can start by selecting the initial view [image]  or the top-down view  [image].

Available named views:
init_view, top_down_view

Step 1/10

ASSISTANT: <think>I'll begin from the initial view to anchor my yaw and local scene layout before moving.</think><action>select_view(init_view)|</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=1.2000, ty=1.8000, tz=2.5000, rx=-90.00°, ry=0.00°, rz=30.00°]
Available views: init_view, top_down_view
[image]

Step 2/10

ASSISTANT: <think>I see the sink ahead-left and a window ahead. The target likely sits further right of the sink axis. I'll yaw right to align with the aisle, then advance and strafe right to open sight to the shower area.</think><action>turn_right|move_forward|move_right</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=1.6500, ty=2.3000, tz=2.5000, rx=-90.00°, ry=0.00°, rz=0.00°]
Available views: init_view, top_down_view
[image]

Step 3/10

ASSISTANT: <think>The curtain rail appears at far-right. I will fine-tune by advancing once and slight left to center the curtain-window pair.</think><action>move_forward|move_left</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=1.4000, ty=2.8000, tz=2.5000, rx=-90.00°, ry=0.00°, rz=0.00°]
Available views: init_view, top_down_view
[image]

Step 4/10

ASSISTANT: <think>The target composition looks matched: window near center, curtain to the right, frontal yaw. I'll answer with the estimated pose.</think><action>answer(1.40, 2.80, 2.50, -90.0, 0.0, 0.0)|</action>
""",

        # ---------------- Example 2 ----------------
        """Example 2:
USER: Now please estimate the camera pose of the target view  .
You can start by selecting the initial view [image]  or the top-down view  [image].

Available named views:
init_view, top_down_view

Step 1/10

ASSISTANT: <think>I will start from the top-down view to anchor global layout and plan a direct path to the shower-corner target.</think><action>select_view(top_down_view)|</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.6000, ty=1.6000, tz=2.9000, rx=-90.00°, ry=0.00°, rz=90.00°]
Available views: init_view, top_down_view
[image]

Step 2/10

ASSISTANT: <think>From top-down, the target sits east of the sink line and slightly north. I'll drop into an approximate aisle-facing pose to begin.</think><action>get_view(2.10,1.90,2.80,-90.0,0.0,30.0)|</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.1000, ty=1.9000, tz=2.8000, rx=-90.00°, ry=0.00°, rz=30.00°]
Available views: init_view, top_down_view
[image]

Step 3/10

ASSISTANT: <think>The shower rail is off to the right. I will yaw right to align, then advance and strafe right to open viewpoint.</think><action>turn_right|move_forward|move_right</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.5500, ty=2.4000, tz=2.8000, rx=-90.00°, ry=0.00°, rz=0.00°]
Available views: init_view, top_down_view
[image]

Step 4/10

ASSISTANT: <think>Good—curtain edge appears but off-center. I'll micro-adjust with a left strafe and a slight yaw left to balance window and curtain.</think><action>move_left|turn_left</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.3000, ty=2.4000, tz=2.8000, rx=-90.00°, ry=0.00°, rz=30.00°]
Available views: init_view, top_down_view
[image]

Step 5/10

ASSISTANT: <think>Vertical framing is a bit high. I will pitch down to place sill and rail in the intended band, then step forward for scale match.</think><action>look_down|move_forward</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.3000, ty=2.9000, tz=2.8000, rx=-60.00°, ry=0.00°, rz=30.00°]
Available views: init_view, top_down_view
[image]

Step 6/10

ASSISTANT: <think>The pitch seems slightly over; restore nominal horizon and center by a tiny right strafe.</think><action>look_up|move_right</action>

USER: Status: ok
Current camera 6-DoF (c2w, Euler XYZ, DEGREES):
[tx=2.5500, ty=2.9000, tz=2.8000, rx=-90.00°, ry=0.00°, rz=30.00°]
Available views: init_view, top_down_view
[image]

Step 7/10

ASSISTANT: <think>Composition matches: window centered, curtain at right third, mild left yaw. I'll finalize the pose estimate.</think><action>answer(2.55, 2.90, 2.80, -90.0, 0.0, 30.0)|</action>
"""
    ]

    example_count = max(0, min(int(example_count), len(examples)))
    return "\n\n".join(examples[:example_count])


def build_system_prompt(
    *,
    tool_instruction: str,
    max_turns: int,
    use_examples: bool = True,
    example_count: int = 2,
    format_name: str = "free_think",
    obs_format_bonus_reward: float = 0.0,
) -> str:
    """
    Build the system prompt string used by GymProxyTool.system_prompt().

    When ``obs_format_bonus_reward > 0``, an additional REASONING BONUS
    section is appended to inform the model that emitting the
    ``<observation>...</observation> ... <action>...</action>`` format
    earns an extra per-turn reward — keeps the SFT-taught reasoning
    habit alive during RL.
    """
    fmt_instr = get_format_instruction(format_name, action_example="action_1|action_2|action_3...")
    preface = (
        "You are solving an active-exploration pose estimation task.\n\n"
        "GOAL\n"
        "Predict the TARGET VIEW absolute camera pose (camera-to-world, c2w) "
        "as a 6-DoF vector:\n"
        "[tx, ty, tz, rx, ry, rz]\n\n"
        "- tx, ty, tz are translations in meters\n"
        "- rx, ry, rz are rotations in DEGREES\n"
        "- rotation order is Euler XYZ\n\n"
        "You may explore the 3D scene using the available camera-control actions, then submit a final answer.\n"
        "Your predicted pose should be as close as possible to the target pose. To achieve this, navigate to a view that matches the target view as closely as possible.\n"
        "TURN LIMIT\n"
        f"You must complete the task within {int(max_turns)} turns, "
        "including the final answer.\n\n"
        f"OUTPUT FORMAT (STRICT)\n"
        f"{fmt_instr}\n\n"
        "FORMAT RULES\n"
        "- Do NOT output any text outside the expected tags.\n"
        "- Use '|' to separate multiple actions.\n"
        "- Actions must be chosen from the supported action list.\n"
        "- The final response MUST contain exactly one answer(...).\n"
        "- The episode terminates immediately after answer(...).\n"
        "- You may explore first, or answer immediately if confident.\n"
    )

    if obs_format_bonus_reward and obs_format_bonus_reward > 0:
        preface += (
            "\nREASONING BONUS\n"
            "In each turn, if your response follows the format "
            "`<observation>...</observation> ... <action>...</action>`, "
            f"you receive an extra bonus reward of {obs_format_bonus_reward}.\n"
        )

    obs_str = (preface + "\n\n" + (tool_instruction or "")).strip()
    if use_examples:
        obs_str = (obs_str + "\n\n" + build_example_prompt(example_count)).strip()
    return obs_str

def build_reset_prompt(question: str, choices: str = "", view_names: List[str] | None = None) -> str:
    """
    Minimal per-episode prompt. Show question, choices, and available views (if any).
    """
    lines: List[str] = []
    if question: lines.append(question.strip())
    if choices:  lines.append(choices.strip())
    if view_names:
        lines += ["", "Available named views:", ", ".join(sorted(view_names))]
    # lines += [
    #     "",
    #     "Return in free_think format: <think>...</think><action>a1|a2|a3</action>",
    #     "You may explore first (e.g., select_view(view_0)), or answer directly if you are confident.",
    # ]
    return "\n".join(lines).strip()
