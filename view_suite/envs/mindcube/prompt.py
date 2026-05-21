# prompt.py  –  MindCube QA environment prompts
# -*- coding: utf-8 -*-
"""
Prompt builders aligned with MindCube's original templates
(src/prompt_generation/templates.py – FF_RSN style).

Response format: <think>...</think><action>answer(x)|</action>
"""
from __future__ import annotations
from typing import List

from view_suite.envs.utils.parse_utils import get_format_instruction

# Matches MindCube QUESTION_HEADER
QUESTION_HEADER = "[Question]\n"


def build_system_prompt(format_name: str = "free_think") -> str:
    fmt_instr = get_format_instruction(format_name, action_example="answer(x)", action_description="where x is A, B, C, or D.")
    return (
        "[Task]\n"
        "Your task is to analyze the spatial arrangement of objects in the scene "
        "by examining the provided images, which show the scene from different viewpoints.\n"
        "[Answer Instruction]\n"
        f"{fmt_instr}"
    )


def build_example_prompt(example_count: int) -> str:
    """
    Return few-shot examples for MindCube QA in free_think format.
    Returns empty string when example_count <= 0.
    """
    if example_count <= 0:
        return ""

    examples = [
        # ---- Example 1 -> answer A ----
        """Example 1:
USER: Based on these four images showing the toy car from different viewpoints (front, left, back, and right), where is the red block relative to the toy car?
A. front
B. right
C. back
D. left

ASSISTANT: <think>Looking at the front view (image 1), I can see the red block directly ahead of the toy car. In the left view (image 2), the red block appears on the right side, confirming it is in front from the original perspective. The back view (image 3) shows the red block is not visible behind the car. Therefore, the red block is in front of the toy car.</think><action>answer(A)|</action>""",

        # ---- Example 2 -> answer B ----
        """Example 2:
USER: Based on these four images showing the cup from different viewpoints (front, left, back, and right), where is the spoon relative to the cup?
A. front
B. right
C. back-left
D. left

ASSISTANT: <think>From the front view (image 1), the spoon is not directly ahead. In the left view (image 2), the spoon appears behind, and in the right view (image 4), the spoon is clearly visible in front. This means from the original front-facing perspective, the spoon is to the right of the cup.</think><action>answer(B)|</action>""",

        # ---- Example 3 -> answer C ----
        """Example 3:
USER: Based on these four images showing the laptop from different viewpoints (front, left, back, and right), where is the mouse relative to the laptop?
A. right
B. back-right
C. back-left
D. front

ASSISTANT: <think>Examining the front view (image 1), the mouse is not directly visible. In the back view (image 3), the mouse appears toward the right side. In the left view (image 2), the mouse appears behind. Combining these observations, the mouse is positioned at the back-left relative to the laptop's front-facing orientation.</think><action>answer(C)|</action>""",

        # ---- Example 4 -> answer D ----
        """Example 4:
USER: Based on these four images showing the bottle from different viewpoints (front, left, back, and right), where is the box relative to the bottle?
A. front-right
B. right
C. back
D. left

ASSISTANT: <think>In the front view (image 1), the box is not visible on the right. In the left view (image 2), the box appears directly in front. In the right view (image 4), the box is behind. These observations consistently indicate the box is to the left of the bottle.</think><action>answer(D)|</action>""",
    ]

    example_count = max(0, min(example_count, len(examples)))
    return "\n\n".join(examples[:example_count])


def build_reset_prompt(question: str, num_images: int = 0) -> str:
    """
    Per-episode prompt shown at reset.

    Layout::

        [Question]
        <image> <image> ...        ← one per loaded image
        Based on these four images ...

    The ``<image>`` placeholders sit between the header and the question
    text so the vision model sees the images before reading the question
    that references them.
    """
    q = (question or "").strip()
    if not q:
        return ""
    parts = [QUESTION_HEADER.rstrip()]
    if num_images > 0:
        parts.append(" ".join(["<image>"] * num_images))
    parts.append(q)
    return "\n".join(parts)
